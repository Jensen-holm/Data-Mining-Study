package request

import (
	"fmt"

	"github.com/go-gota/gota/dataframe"
	"github.com/gofiber/fiber/v2"
)

type Payload struct {
	CSVData   string   `json:"csv_data"`
	Features  []string `json:"features"`
	Target    string   `json:"target"`
	Algorithm string   `json:"algorithm"`

	Args map[string]interface{} `json:"args"`
	Df   dataframe.DataFrame
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
