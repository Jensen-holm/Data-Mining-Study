package main

import (
	"strings"

	"github.com/Jensen-holm/ml-from-scratch/alg"
	"github.com/Jensen-holm/ml-from-scratch/nn"
	"github.com/Jensen-holm/ml-from-scratch/request"
	"github.com/go-gota/gota/dataframe"
	"github.com/gofiber/fiber/v2"
)

func main() {
	app := fiber.New()

	algMap := map[string]alg.Alg{
		"neural_network": nn.NN,
	}

	app.Post("/", func(c *fiber.Ctx) error {
		r := new(request.Payload)

		err := request.NewPayload(r, c)
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "Invalid JSON data",
			})
		}

		df := dataframe.ReadCSV(strings.NewReader(r.CSVData))
		r.SetDf(df)

		a := algMap[r.Algorithm]
		a.New(r)

		return c.SendString("No error")
	})

	app.Listen(":3000")

}
