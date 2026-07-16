use std::env;

use wwama::{Error, Result, Session, SessionOptions};

fn main() -> Result<()> {
    let path = env::args().nth(1).ok_or(Error::InvalidInput)?;
    let options = SessionOptions {
        n_gpu_layers: 0,
        ..SessionOptions::default()
    };
    let session = Session::load_from_path(&path, options)?;
    for tensor in session.model().tensors()? {
        println!(
            "{} type={}({}) dims={:?} strides={:?} bytes={} backend={}",
            tensor.name,
            tensor.type_name,
            tensor.type_id,
            &tensor.dimensions[..tensor.n_dims],
            &tensor.strides[..tensor.n_dims],
            tensor.nbytes,
            tensor.backend,
        );
    }
    Ok(())
}
