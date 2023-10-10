package nn

import (
	"fmt"
	"strings"

	"github.com/go-gota/gota/dataframe"
	"github.com/gofiber/fiber/v2"
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

	// iterate n times where n = nn.Epochs
	// use backprop algorithm on each iteration
	// to fit the model to the data

}
