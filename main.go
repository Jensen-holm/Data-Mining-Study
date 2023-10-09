package main

import (
	"fmt"
	"os"

	"github.com/go-gota/gota/dataframe"
	"github.com/gofiber/fiber/v2"
)

func main() {
	app := fiber.New()

	app.Get("/", func(c *fiber.Ctx) error {
		filePath := "test/iris.csv"

		file, err := os.Open(filePath)
		if err != nil {
			panic(err)
		}

		df := dataframe.ReadCSV(file)
		fmt.Println(df)
		return c.SendString("No error")
	})

	app.Listen(":3000")

}
