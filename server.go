package main

import (
	"fmt"

	"github.com/Jensen-holm/ml-from-scratch/nn"
	"github.com/gofiber/fiber/v2"
)

func main() {
	app := fiber.New()

	// eventually we might want to add a key to this endpoint
	// that we will be able to validate.
	app.Post("/neural-network", func(c *fiber.Ctx) error {

		nn, err := nn.NewNN(c)
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": err,
			})
		}

		nn.Train()
		fmt.Println(nn.Wo)

		return c.SendString("No error")
	})

	app.Listen(":3000")
}
