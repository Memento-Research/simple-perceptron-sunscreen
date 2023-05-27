
struct SimplePerceptron{
    weights: Vec<f64>,
    bias: f64,
}

impl SimplePerceptron{

    fn new(input_size: usize) -> SimplePerceptron{
        let weights = vec![0.0; input_size];
        let bias = 0.0;

        SimplePerceptron{
            weights,
            bias,
        }
    }

    fn predict(&self, inputs: &Vec<f64>) -> f64{
        let mut sum = 0.0;
        for i in 0..inputs.len(){
            sum += inputs[i] * self.weights[i];
        }
        self.activation(sum + self.bias)
    }

    fn train(&mut self, inputs: &Vec<Vec<f64>>, outputs: &Vec<f64>, epochs: usize, learning_rate: f64){
        for _ in 0..epochs{
            for i in 0..inputs.len(){
                let prediction = self.predict(&inputs[i]);
                let error = outputs[i] - prediction;
                for j in 0..self.weights.len(){
                    self.weights[j] += error * inputs[i][j] * learning_rate;
                }
                self.bias += error * learning_rate;
            }
        }
    }

    fn activation(&self, x: f64) -> f64{
        if x >= 0.0 {
            1.0
        } else {
            0.0
        }
    }
}




fn main(){

    let mut perceptron = SimplePerceptron::new(2);

    // OR Gate
    let inputs = vec![vec![-1.0, -1.0], vec![-1.0, 1.0], vec![1.0, -1.0], vec![1.0, 1.0]];
    let outputs = vec![-1.0, 1.0, 1.0, 1.0];

    perceptron.train(&inputs, &outputs, 100000, 0.0001);

    println!("OR Gate");
    println!("0 or 0 = {}", perceptron.predict(&vec![-1.0, -1.0]));
    println!("0 or 1 = {}", perceptron.predict(&vec![-1.0, 1.0]));
    println!("1 or 0 = {}", perceptron.predict(&vec![1.0, -1.0]));
    println!("1 or 1 = {}", perceptron.predict(&vec![1.0, 1.0]));
}

