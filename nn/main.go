package nn

import (
	"fmt"
	"math/rand"
	"strings"

	"github.com/go-gota/gota/dataframe"
	"github.com/gofiber/fiber/v2"
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
	XTrain         *dataframe.DataFrame
	YTrain         *dataframe.DataFrame
	XTest          *dataframe.DataFrame
	YTest          *dataframe.DataFrame
	Wh             *[][]float64
	Bh             *[]float64
	Wo             *[][]float64
	Bo             *[]float64
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

	// input hidden layer weights
	wh := make([][]float64, inputSize)
	for i := range wh {
		wh[i] = make([]float64, hiddenSize)
		for j := range wh[i] {
			wh[i][j] = rand.Float64() - 0.5
		}
	}

	bh := make([]float64, hiddenSize)
	for i := range bh {
		bh[i] = rand.Float64() - 0.5
	}

	// initialize weights and biases for hidden -> output layer
	wo := make([][]float64, hiddenSize)
	for i := range wo {
		wo[i] = make([]float64, outputSize)
		for j := range wo[i] {
			wo[i][j] = rand.Float64() - 0.5
		}
	}

	bo := make([]float64, outputSize)
	for i := range bo {
		bo[i] = rand.Float64() - 0.5
	}

	nn.Wh = &wh
	nn.Bh = &bh
	nn.Wo = &wo
	nn.Bo = &bo
}
