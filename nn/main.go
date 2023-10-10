package nn

import (
	"fmt"
	"math/rand"
	"strings"

	"github.com/go-gota/gota/dataframe"
	"github.com/gofiber/fiber/v2"
	"gonum.org/v1/gonum/mat"
)

type NN struct {
	CSVData        string   `json:"csv_data"`
	Features       []string `json:"features"`
	Target         string   `json:"target"`
	Epochs         int      `json:"epochs"`
	HiddenSize     int      `json:"hidden_size"`
	LearningRate   float64  `json:"learning_rate"`
	ActivationFunc string   `json:"activation"`
	TestSize       float64  `json:"test_size"`

	Df     *dataframe.DataFrame
	XTrain dataframe.DataFrame
	YTrain dataframe.DataFrame
	XTest  dataframe.DataFrame
	YTest  dataframe.DataFrame
}

func NewNN(c *fiber.Ctx) (*NN, error) {
	newNN := new(NN)
	err := c.BodyParser(newNN)
	if err != nil {
		return nil, fmt.Errorf("invalid JSON data: %v", err)
	}
	df := dataframe.ReadCSV(strings.NewReader(newNN.CSVData))
	newNN.Df = &df
	return newNN, nil
}

func (nn *NN) Train() {
	// train test split the data
	XTrain, XTest, YTrain, YTest := nn.trainTestSplit()

	weights, biases := nn.InitWnB()

	// iterate n times where n = nn.Epochs
	// use backprop algorithm on each iteration
	// to fit the model to the data

}

func (nn *NN) InitWnB() {
	// randomly initialize weights and biases to start
	inputSize := len(nn.Features)
	hiddenSize := nn.HiddenSize
	outputSize := 1 // only predicting one thing for now

	// Initialize weights and biases for the input layer to hidden layer
	weightsInputHidden := mat.NewDense(inputSize, hiddenSize, nil)
	weightsInputHidden.Apply(func(_, _ int, v float64) float64 {
		// Randomly initialize weights with values between -1 and 1
		return rand.Float64()*2 - 1
	}, weightsInputHidden)

	biasesHidden := mat.NewVecDense(hiddenSize, nil)
	biasesHidden.Apply(func(_, _ int, v float64) float64 {
		// Randomly initialize biases
		return rand.Float64()
	}, biasesHidden)

	// Initialize weights and biases for the hidden layer to output layer
	weightsHiddenOutput := mat.NewDense(hiddenSize, outputSize, nil)
	weightsHiddenOutput.Apply(func(_, _ int, v float64) float64 {
		// Randomly initialize weights with values between -1 and 1
		return rand.Float64()*2 - 1
	}, weightsHiddenOutput)

	biasesOutput := mat.NewVecDense(outputSize, nil)
	biasesOutput.Apply(func(_, _ int, v float64) float64 {
		// Randomly initialize biases
		return rand.Float64()
	}, biasesOutput)
}
