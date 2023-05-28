use std::ops::{Add, AddAssign, Mul, Sub};

use sunscreen::{
    fhe_program,
    types::{bfv::Fractional, Cipher},
    Compiler, FheProgramInput, Runtime,
};

struct SimplePerceptron {
    weights: Vec<f64>,
    bias: f64,
}

#[fhe_program(scheme = "bfv")]
fn predict(
    weights: [Fractional<64>; 2],
    inputs: [Cipher<Fractional<64>>; 2],
    bias: Fractional<64>,
) -> Cipher<Fractional<64>> {
    let sum = inputs[0] * weights[0] + inputs[1] * weights[1] + bias;
    sum
}

impl SimplePerceptron {
    fn new(input_size: usize) -> SimplePerceptron {
        let weights = vec![0.0; input_size];
        let bias = 0.0;

        SimplePerceptron { weights, bias }
    }

    fn predict_impl<T>(&self, inputs: &[T; 2]) -> T
    where
        T: Mul<f64, Output = T>
            + Add<f64, Output = T>
            + Add<T, Output = T>
            + Sub<Output = T>
            + AddAssign<T>
            + Copy,
    {
        let sum: T = inputs[0] * self.weights[0] + inputs[1] * self.weights[1] + self.bias;
        self.activation_impl(sum)
    }

    fn train(
        &mut self,
        inputs: &Vec<[f64; 2]>,
        outputs: &Vec<f64>,
        epochs: usize,
        learning_rate: f64,
    ) {
        for _ in 0..epochs {
            for i in 0..inputs.len() {
                let prediction = self.predict_impl(&inputs[i]);
                let error = outputs[i] - prediction;
                for j in 0..self.weights.len() {
                    self.weights[j] += error * inputs[i][j] * learning_rate;
                }
                self.bias += error * learning_rate;
            }
        }
    }

    fn activation_impl<T>(&self, x: T) -> T {
        x
    }

    fn get_weights(&self) -> [Fractional<64>; 2] {
        [self.weights[0].into(), self.weights[1].into()]
    }

    fn get_bias(&self) -> Fractional<64> {
        self.bias.into()
    }
}

fn main() {
    let app = Compiler::new().fhe_program(predict).compile().unwrap();
    let runtime = Runtime::new(app.params()).unwrap();
    let (public_key, private_key) = runtime.generate_keys().unwrap();

    let mut perceptron = SimplePerceptron::new(2);

    // OR Gate
    let inputs = vec![[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]];
    let outputs = vec![-1.0, 1.0, 1.0, 1.0];

    perceptron.train(&inputs, &outputs, 100000, 0.01);

    println!("OR Gate");
    println!("0 or 0 = {}", perceptron.predict_impl(&[-1.0, -1.0]));
    println!("0 or 1 = {}", perceptron.predict_impl(&[-1.0, 1.0]));
    println!("1 or 0 = {}", perceptron.predict_impl(&[1.0, -1.0]));
    println!("1 or 1 = {}", perceptron.predict_impl(&[1.0, 1.0]));

    let weights = perceptron.get_weights();
    let bias = perceptron.get_bias();

    let encrypted_input_a = Fractional::<64>::from(-1.0);
    let encrypted_input_b = Fractional::<64>::from(-1.0);
    let encrypted_input = runtime
        .encrypt([encrypted_input_a, encrypted_input_b], &public_key)
        .unwrap();

    let args: Vec<FheProgramInput> = vec![weights.into(), encrypted_input.into(), bias.into()];
    let results = runtime
        .run(app.get_program(predict).unwrap(), args, &public_key)
        .unwrap();
    let decrypted_result: Fractional<64> = runtime.decrypt(&results[0], &private_key).unwrap();
    assert_eq!(
        decrypted_result,
        Fractional::<64>::from(perceptron.predict_impl(&[-1.0, -1.0]))
    );
    let result_f64: f64 = decrypted_result.into();
    println!("FHE: 0 or 0 = {}", result_f64);
}
