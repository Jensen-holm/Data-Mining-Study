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
	LearningRate   float64  `json:"learning_rate"`
	HiddenSize     int      `json:"hidden_size"`
	ActivationFunc string   `json:"activation"`
}

func main() {
	csvBytes, err := ioutil.ReadFile("iris.csv")
	if err != nil {
		fmt.Println("Error reading CSV file: ", err)
		return
	}

	csvString := string(csvBytes)
	target := "species"
	features := []string{
		"petal length",
		"sepal length",
		"sepal width",
		"petal width",
	}

	payload := RequestPayload{
		CSVData:        csvString,
		Features:       features,
		Target:         target,
		Epochs:         100,
		LearningRate:   0.01,
		HiddenSize:     12,
		ActivationFunc: "tanh",
	}

	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		panic(err)
	}

	r, err := http.Post(
		"http://127.0.0.1:3000/neural-network",
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
