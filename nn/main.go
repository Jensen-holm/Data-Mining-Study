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
	// attributes set by request
	CSVData      string   `json:"csv_data"`
	Features     []string `json:"features"`
	Target       string   `json:"target"`
	Epochs       int      `json:"epochs"`
	HiddenSize   int      `json:"hidden_size"`
	LearningRate float64  `json:"learning_rate"`
	Activation   string   `json:"activation"`
	TestSize     float64  `json:"test_size"`

	// attributes set after args above are parsed
	ActivationFunc *func(float64) float64
	Df             *dataframe.DataFrame
	XTrain         *mat.Dense
	YTrain         *mat.Dense
	XTest          *mat.Dense
	YTest          *mat.Dense
	Wh             *mat.Dense
	Bh             *mat.Dense
	Wo             *mat.Dense
	Bo             *mat.Dense
}

func NewNN(c *fiber.Ctx) (*NN, error) {
	newNN := new(NN)
	err := c.BodyParser(newNN)
	if err != nil {
		return nil, fmt.Errorf("invalid JSON data: %v", err)
	}
	df := dataframe.ReadCSV(strings.NewReader(newNN.CSVData))
	activation := ActivationMap[newNN.Activation]

	newNN.Df = &df
	newNN.ActivationFunc = &activation
	return newNN, nil
}

func (nn *NN) InitWnB() {
	// randomly initialize weights and biases to start
	inputSize := len(nn.Features)
	hiddenSize := nn.HiddenSize
	outputSize := 1 // only predicting one thing

	// Initialize input hidden layer weights as a Gonum matrix
	wh := mat.NewDense(inputSize, hiddenSize, nil)
	wh.Apply(func(i, j int, v float64) float64 {
		return rand.Float64() - 0.5
	}, wh)

	// Initialize hidden layer bias as a Gonum matrix
	bh := mat.NewDense(1, hiddenSize, nil)
	bh.Apply(func(i, j int, v float64) float64 {
		return rand.Float64() - 0.5
	}, bh)

	// Initialize weights and biases for hidden -> output layer as Gonum matrices
	wo := mat.NewDense(hiddenSize, outputSize, nil)
	wo.Apply(func(i, j int, v float64) float64 {
		return rand.Float64() - 0.5
	}, wo)

	bo := mat.NewDense(1, outputSize, nil)
	bo.Apply(func(i, j int, v float64) float64 {
		return rand.Float64() - 0.5
	}, bo)

	nn.Wh = wh
	nn.Bh = bh
	nn.Wo = wo
	nn.Bo = bo
}
