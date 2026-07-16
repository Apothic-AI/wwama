use std::collections::BTreeMap;
use std::env;

use wwama::{GenerationOptions, Result as WwamaResult, Session, SessionOptions, llama_token};

struct MiyagiBackend {
    session: Session,
    projections: BTreeMap<(usize, String), String>,
}

impl MiyagiBackend {
    fn load(path: &str) -> WwamaResult<Self> {
        let options = SessionOptions {
            n_gpu_layers: 0,
            mutable_tensors: true,
            ..SessionOptions::default()
        };
        Ok(Self {
            session: Session::load_from_path(path, options)?,
            projections: BTreeMap::new(),
        })
    }

    fn map_projection(&mut self, layer: usize, projection: &str, tensor_name: &str) {
        self.projections
            .insert((layer, projection.to_owned()), tensor_name.to_owned());
    }

    fn tensor_name(&self, layer: usize, projection: &str) -> WwamaResult<&str> {
        self.projections
            .get(&(layer, projection.to_owned()))
            .map(String::as_str)
            .ok_or(wwama::Error::TensorNotFound)
    }

    fn num_rows(&self, layer: usize, projection: &str) -> WwamaResult<usize> {
        self.session
            .model()
            .tensor(self.tensor_name(layer, projection)?)?
            .row_count()
    }

    fn row_scales(&mut self, layer: usize, projection: &str) -> WwamaResult<Vec<f32>> {
        let name = self.tensor_name(layer, projection)?.to_owned();
        self.session.q1_0_row_scales(&name)
    }

    fn encode(&self, text: &str) -> WwamaResult<Vec<llama_token>> {
        self.session.tokenize_text(text, false, true)
    }

    fn encode_token(&self, text: &str) -> WwamaResult<llama_token> {
        self.encode(text)?
            .last()
            .copied()
            .ok_or(wwama::Error::InvalidInput)
    }

    fn logit_gap(
        &mut self,
        prompt_tokens: &[llama_token],
        correct: llama_token,
        wrong: llama_token,
    ) -> WwamaResult<f32> {
        self.session.logit_gap(prompt_tokens, correct, wrong)
    }

    fn flip_row(&mut self, layer: usize, projection: &str, row: usize) -> WwamaResult<()> {
        let name = self.tensor_name(layer, projection)?.to_owned();
        self.session.xor_q1_0_row(&name, row)?;
        Ok(())
    }

    fn generate(&mut self, prompt: &str, max_tokens: usize) -> WwamaResult<String> {
        let options = GenerationOptions {
            max_new_tokens: max_tokens,
            ..GenerationOptions::default()
        };
        Ok(self.session.generate_text(prompt, &options)?.text)
    }
}

fn main() -> WwamaResult<()> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 7 {
        eprintln!("usage: miyagi_backend MODEL.gguf TENSOR ROW PROMPT CORRECT_TOKEN WRONG_TOKEN");
        return Err(wwama::Error::InvalidInput);
    }
    let mut backend = MiyagiBackend::load(&args[1])?;
    backend.map_projection(0, "candidate", &args[2]);
    let row = args[3]
        .parse::<usize>()
        .map_err(|_| wwama::Error::InvalidInput)?;
    let prompt = backend.encode(&args[4])?;
    let correct = backend.encode_token(&args[5])?;
    let wrong = backend.encode_token(&args[6])?;
    let rows = backend.num_rows(0, "candidate")?;
    let scales = backend.row_scales(0, "candidate")?;
    let baseline = backend.logit_gap(&prompt, correct, wrong)?;
    backend.flip_row(0, "candidate", row)?;
    let mutated = backend.logit_gap(&prompt, correct, wrong)?;
    backend.flip_row(0, "candidate", row)?;
    let restored = backend.logit_gap(&prompt, correct, wrong)?;
    println!(
        "rows={rows} row_scale={} baseline={baseline} mutated={mutated} restored={restored}",
        scales[row]
    );
    let _ = backend.generate(&args[4], 1)?;
    Ok(())
}
