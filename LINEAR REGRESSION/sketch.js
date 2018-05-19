var X = []
var Y = []

var M, B

const m = tf.variable(tf.scalar(Math.random()))
const b = tf.variable(tf.scalar(Math.random()))

const learningRate = 0.5
const optimizer = tf.train.sgd(learningRate)

function setup () {
  createCanvas(windowWidth, windowHeight)
}

function draw () {
  background(51)
  intro()
  plotData()

  if (X.length) {
    tf.tidy(() => {
      const xs = tf.tensor(X, [X.length, 1])
      const ys = tf.tensor(Y, [Y.length, 1])

      train(xs, ys)

      M = m.dataSync()[0]
      B = b.dataSync()[0]
    });
    drawLine()
  }

  // Check Memory Leak
  console.log(tf.memory().numTensors);
}

function predict(x) {
  // y = m * x + b
  return m.mul(x).add(b)
}

function intro () {
  let instruction = "Tap on the Screen to Insert Data Points..."
  fill(250)
  noStroke()
  textFont('monospace')
  textSize(25)
  text("Tensorflow.js : Linear Regression", 15, 40)
  textSize(20)
  text(instruction, 15, windowHeight - 30)
  fill(100)
  textSize(15)
  text("Author : Kaustubh Olpadkar", windowWidth - 270, 40)
  noFill();
  noStroke();
}

function loss(predictions, labels) {
  // Mean Squared Error
  return predictions.sub(labels).square().mean();
}

function train(xs, ys, numIterations = 1) {
  for (let iter = 0; iter < numIterations; iter++) {
    optimizer.minimize(() => loss(predict(xs), ys));
  }
}

function mouseClicked () {

  let normX = map(mouseX, 0, width, 0, 1)
  let normY = map(mouseY, 0, height, 0, 1)

  X.push(normX);
  Y.push(normY);
}

function h (x) {
  return B + M * x;
}

function drawLine () {
  let x1 = 0.0
  let y1 = h(x1);
  let x2 = 1.0
  let y2 = h(x2);

  let denormX1 = Math.floor(map(x1, 0, 1, 0, width))
  let denormY1 = Math.floor(map(y1, 0, 1, 0, height))
  let denormX2 = Math.floor(map(x2, 0, 1, 0, width))
  let denormY2 = Math.floor(map(y2, 0, 1, 0, height))

  stroke(255);
  line(denormX1, denormY1, denormX2, denormY2);
}

function plotData () {
  noStroke();
  fill(255);
  for (let i = 0; i < X.length; i++) {
    let denormX = Math.floor(map(X[i], 0, 1, 0, width))
    let denormY = Math.floor(map(Y[i], 0, 1, 0, height))
    ellipse(denormX, denormY, 10);
  }
  noFill();
}
