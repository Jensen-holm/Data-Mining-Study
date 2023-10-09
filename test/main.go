package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
)

func main() {

	// in this script, we are going to test
	// the API endpoint for the neural network
	r, err := http.Get("http://127.0.0.1:3000/")
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
