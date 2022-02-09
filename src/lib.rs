use pyo3::prelude::*;

pub mod utils_rust;
pub mod spacetime;
pub mod collision;
pub mod core;
pub mod optimization;


#[pymodule]
fn cairo_planning_core(py: Python, m: &PyModule) -> PyResult<()> {
    core::agents::register(py, m)?;
    Ok(())
}


