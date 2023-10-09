package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
)

type RequestPayload struct {
	CSVData   string   `json:"csv_data"`
	Features  []string `json:"features"`
	Target    string   `json:"target"`
	Algorithm string   `json:"algorithm"`
	Args      map[string]interface{}
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
	args := map[string]interface{}{
		"epochs":        100,
		"hidden_size":   8,
		"learning_rate": 0.1,
		"activation":    "tanh",
	}

	payload := RequestPayload{
		CSVData:  csvString,
		Features: features,
		Target:   target,
		Args:     args,
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
