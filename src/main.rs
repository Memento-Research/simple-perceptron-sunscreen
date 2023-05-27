use std::ops::{Add, Mul, Sub, AddAssign};

struct SimplePerceptron {
    weights: Vec<f64>,
    bias: f64,
}

impl SimplePerceptron {
    fn new(input_size: usize) -> SimplePerceptron {
        let weights = vec![0.0; input_size];
        let bias = 0.0;

        SimplePerceptron { weights, bias }
    }

    fn predict<T>(&self, inputs: &[T; 2]) -> T
    where
        T: Mul<f64, Output = T> + Add<f64, Output = T> + Add<T, Output = T> + Sub<Output = T> + AddAssign<T> + Copy,
    {
        let sum: T = inputs[0] * self.weights[0] + inputs[1] * self.weights[1] + self.bias;
        self.activation(sum)
    }

    fn train(
        &mut self,
        inputs: &Vec<[f64; 2]>,
        outputs: &Vec<f64>,
        epochs: usize,
        learning_rate: f64,
    )
    {
        for _ in 0..epochs {
            for i in 0..inputs.len() {
                let prediction = self.predict(&inputs[i]);
                let error = outputs[i] - prediction;
                for j in 0..self.weights.len() {
                    self.weights[j] += error * inputs[i][j] * learning_rate;
                }
                self.bias += error * learning_rate;
            }
        }
    }

    fn activation<T>(&self, x: T) -> T
    {
        x
    }
}

fn main() {
    let mut perceptron = SimplePerceptron::new(2);

    // OR Gate
    let inputs = vec![[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]];
    let outputs = vec![-1.0, 1.0, 1.0, 1.0];

    perceptron.train(&inputs, &outputs, 100000, 0.01);

    println!("OR Gate");
    println!("0 or 0 = {}", perceptron.predict(&[-1.0, -1.0]));
    println!("0 or 1 = {}", perceptron.predict(&[-1.0, 1.0]));
    println!("1 or 0 = {}", perceptron.predict(&[1.0, -1.0]));
    println!("1 or 1 = {}", perceptron.predict(&[1.0, 1.0]));
}
