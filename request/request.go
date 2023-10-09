package request

import (
	"fmt"

	"github.com/go-gota/gota/dataframe"
	"github.com/gofiber/fiber/v2"
)

type Payload struct {
	CSVData        string   `json:"csv_data"`
	Features       []string `json:"features"`
	Target         string   `json:"target"`
	Epochs         int      `json:"epochs"`
	HiddenSize     int      `json:"hidden_size"`
	LearningRate   float64  `json:"learning_rate"`
	ActivationFunc string   `json:"activation_func"`

	Df dataframe.DataFrame
}

func (p *Payload) SetDf(df dataframe.DataFrame) {
	p.Df = df
}

func NewPayload(dest *Payload, c *fiber.Ctx) error {
	if err := c.BodyParser(dest); err != nil {
		return fmt.Errorf("invalid JSON data")
	}
	return nil
}
