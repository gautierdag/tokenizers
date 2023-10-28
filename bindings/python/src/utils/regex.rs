use fancy_regex::Regex;
use pyo3::exceptions;
use pyo3::prelude::*;

/// Instantiate a new Regex with the given pattern
#[pyclass(module = "tokenizers", name = "Regex")]
pub struct PyRegex {
    pub inner: Regex,
    pub pattern: String,
}

#[pymethods]
impl PyRegex {
    #[new]
    #[pyo3(text_signature = "(self, pattern)")]
    fn new(s: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Regex::new(s)
                .map_err(|e| exceptions::PyException::new_err(e.to_string()))?,
            pattern: s.to_owned(),
        })
    }
}