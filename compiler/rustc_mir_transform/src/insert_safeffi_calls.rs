//! SafeFFI instrumentation pass.
//!
//! This pass is added to the
//! pipeline right after [`super::run_runtime_lowering_passes`] in
//! [`super::run_analysis_to_runtime_passes`]) if `-Zsafeffi` is enabled.

use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::{
    Body, BorrowKind, LocalDecls, Location, Place, ProjectionElem, RawPtrKind, Rvalue, Statement,
    StatementKind,
};
use rustc_middle::ty::{Region, Ty, TyCtxt, TypingEnv};
use tracing::debug;

pub(super) struct InsertSafeFfiCalls;

impl<'tcx> crate::MirPass<'tcx> for InsertSafeFfiCalls {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let typing_env = body.typing_env(tcx);
        let mut visitor = SafeFfiCastVisitor {
            tcx,
            local_decls: &body.local_decls,
            typing_env,
            cast_sites: Vec::new(),
        };
        for (bb, data) in body.basic_blocks.as_mut_preserves_cfg().iter_enumerated_mut() {
            visitor.visit_basic_block_data(bb, data);
        }

        let patch = crate::patch::MirPatch::new(body);
        for cast in visitor.cast_sites {
            let _span = patch.source_info_for_location(body, cast.location).span;
            // FIXME: how to actually call the intrinsic? Via NonDivergingIntrinsic??
            /*
            let requirements = SafeFFISafePointerRequirements {
                raw_ptr: cast.ptr,
                size: cast.size,
                align: cast.align,
            };
            patch.add_statement(
                location,
                StatementKind::Intrinsic(Box::new(NonDivergingIntrinsic::SafeFFICheck(
                    SafeFFICheckType::RawToSafeCast(requirements),
                ))),
            );*/
        }
        patch.apply(body);
    }

    fn is_required(&self) -> bool {
        // Instrumentation: not required for correctness; gated on `-Zsafeffi`.
        false
    }
}

// Detection of SafeFFI pointer casts in MIR rvalues.
//
// - `RawToSafe`: `&(*raw_ptr)` — produces a safe reference from a raw pointer
// - `SafeToRaw`: `&raw const x` / `&raw mut x` — produces a raw pointer from
//   a safe reference

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(super) enum SafeFfiCastKind<'tcx> {
    /// `&(*raw_ptr)` — raw pointer dereferenced and re-borrowed as a safe ref.
    RawToSafe { region: Region<'tcx>, borrow_kind: BorrowKind, raw_ptr: Place<'tcx> },
    /// `&raw const x` / `&raw mut x` — safe reference taken as a raw pointer.
    SafeToRaw { raw_ptr_kind: RawPtrKind },
}

/// Inspect an rvalue and report whether it is a SafeFFI cast.
pub(super) fn is_safeffi_cast<'tcx>(
    rvalue: &Rvalue<'tcx>,
    local_decls: &LocalDecls<'tcx>,
) -> Option<SafeFfiCastKind<'tcx>> {
    match rvalue {
        Rvalue::Ref(mir_region, borrow_kind, place) => {
            let ptr_ty = deref_pointer_type(place, local_decls)?;
            if ptr_ty.is_raw_ptr() {
                Some(SafeFfiCastKind::RawToSafe {
                    region: *mir_region,
                    borrow_kind: *borrow_kind,
                    raw_ptr: *place,
                })
            } else {
                None
            }
        }
        Rvalue::RawPtr(raw_ptr_kind, _place) => {
            Some(SafeFfiCastKind::SafeToRaw { raw_ptr_kind: *raw_ptr_kind })
        }
        _ => None,
    }
}

/// We expect `place` to be `(*raw_ptr)`, i.e. it ends in a `Deref` projection.
/// `place.ty()` would give the type *after* that deref (the
/// pointee `T`); we want the type of the pointer itself
/// (`*mut T` / `*const T`), which is the type of the place one
/// projection level up.
fn deref_pointer_type<'tcx>(
    place: &Place<'tcx>,
    local_decls: &LocalDecls<'tcx>,
) -> Option<Ty<'tcx>> {
    let mut iter = place.projection.iter().rev();
    if !matches!(iter.next(), Some(ProjectionElem::Deref)) {
        return None;
    }
    for proj in iter {
        //FIXME: there might be more projections that we need to handle here like `Index` or `ConstantIndex`.
        if let ProjectionElem::Field(_, ty) = proj {
            return Some(ty);
        }
    }
    Some(local_decls[place.local].ty)
}

/// A detected `RawToSafe` cast site, together with the pointer safety
/// requirements derived from the assignment's left-hand-side type (i.e. the
/// safe-reference type the raw pointer is being cast to).
#[allow(dead_code)]
struct SafeFfiCastSite<'tcx> {
    location: Location,
    ptr: Place<'tcx>,
    size: u64,
    align: u64,
}

/// Walks a body's statements looking for SafeFFI casts, recording the
/// location and pointer safety requirements of each one so the pass can
/// insert a runtime check before it.
struct SafeFfiCastVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    local_decls: &'a LocalDecls<'tcx>,
    typing_env: TypingEnv<'tcx>,
    cast_sites: Vec<SafeFfiCastSite<'tcx>>,
}

impl<'tcx> MutVisitor<'tcx> for SafeFfiCastVisitor<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        if let StatementKind::Assign(box (place, rvalue)) = &statement.kind {
            // The left-hand side of the assignment is the safe reference that
            // would be produced by a `RawToSafe` cast, e.g.
            // `place: &'a T = &(*raw_ptr)`. Extract `T`'s size/align up front,
            // before even checking whether this is a SafeFFI cast.
            let place_ty = place.ty(self.local_decls, self.tcx).ty;
            let layout = place_ty.builtin_deref(true).and_then(|referent_ty| {
                self.tcx.layout_of(self.typing_env.as_query_input(referent_ty)).ok()
            });

            if let Some(SafeFfiCastKind::RawToSafe { raw_ptr, .. }) =
                is_safeffi_cast(rvalue, self.local_decls)
            {
                match layout {
                    Some(layout) => {
                        let size = layout.size.bytes();
                        let align = layout.align.abi.bytes();
                        debug!(
                            "[safeffi] cast detected at {:?}: raw_ptr={:?} referent size={} align={}",
                            location, raw_ptr, size, align
                        );
                        self.tcx.dcx().span_note(
                            statement.source_info.span,
                            format!(
                                "[safeffi] SafeFFI `RawToSafe` cast detected: raw_ptr={raw_ptr:?}, \
                                 referent size={size}, align={align}"
                            ),
                        );
                        self.cast_sites.push(SafeFfiCastSite {
                            location,
                            ptr: raw_ptr,
                            size,
                            align,
                        });
                    }
                    None => {
                        self.tcx.dcx().span_warn(
                            statement.source_info.span,
                            format!(
                                "[safeffi] could not determine size/align for `{place_ty}`; \
                                 skipping instrumentation for this cast"
                            ),
                        );
                    }
                }
            }
        }

        self.super_statement(statement, location);
    }
}
