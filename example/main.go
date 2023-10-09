package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
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
	filePath := "iris.csv"

	csvBytes, err := ioutil.ReadFile(filePath)
	if err != nil {
		fmt.Println("Error reading CSV file: ", err)
		return
	}

	csvString := string(csvBytes)
	features := []string{"petal length", "sepal length", "sepal width", "petal width"}
	target := "species"
	epochs := 100
	hiddenSize := 8
	learningRate := 0.1
	activationFunc := "tanh"

	payload := RequestPayload{
		CSVData:        csvString,
		Features:       features,
		Target:         target,
		Epochs:         epochs,
		HiddenSize:     hiddenSize,
		LearningRate:   learningRate,
		ActivationFunc: activationFunc,
	}

	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		panic(err)
	}

	r, err := http.Post(
		"http://127.0.0.1:3000/",
		"application/json",
		bytes.NewBuffer(jsonPayload),
	)
	if err != nil {
		panic(err)
	}

	defer r.Body.Close()

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		panic(err)
	}

	fmt.Println(string(body))

}
