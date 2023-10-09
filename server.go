package main

import (
	"fmt"

	"github.com/gofiber/fiber/v2"
)

type RequestPayload struct {
	CSVData        string   `json:"csv_data"`
	Features       []string `json:"features"`
	Target         string   `json:"target"`
	Epochs         int      `json:"epochs"`
	HiddenSize     int      `json:"hidden_size"`
	LearningRate   float64  `json:"learning_rate"`
	ActivationFunc string   `json:"activation_func"`
}

func main() {
	app := fiber.New()

	app.Post("/", func(c *fiber.Ctx) error {

		requestData := new(RequestPayload)

		// parse json request data into requestData struct
		if err := c.BodyParser(requestData); err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "Invalid JSON data",
			})
		}

		fmt.Println(requestData)

		return c.SendString("No error")
	})

	app.Listen(":3000")

}
