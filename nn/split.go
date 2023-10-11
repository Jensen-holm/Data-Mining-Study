package nn

import (
	"math"
	"math/rand"
)

func (nn *NN) TrainTestSplit() {
	// now we split the data into training
	// and testing based on user specified
	// nn.TestSize.
	nRows := nn.Df.Nrow()
	testRows := int(math.Floor(float64(nRows) * nn.TestSize))

	// subset the testing data
	// randomly select trainRows number of rows
	randStrt := rand.Intn(int(math.Floor(float64(nRows) * nn.TestSize)))
	test := nn.Df.Subset([]int{randStrt, randStrt + testRows})

	// use what is left for training
	allIndices := make([]int, nRows)
	for i := range allIndices {
		allIndices[i] = i
	}

	// Remove the test indices using slice append and variadic parameter
	trainIndices := append(allIndices[:randStrt], allIndices[randStrt+testRows:]...)

	// Create the train DataFrame using the trainIndices
	train := nn.Df.Subset(trainIndices)

	XTrain := train.Select(nn.Features)
	YTrain := train.Select(nn.Target)
	XTest := test.Select(nn.Features)
	YTest := test.Select(nn.Target)

	nn.XTrain = &XTrain
	nn.YTrain = &YTrain
	nn.XTest = &XTest
	nn.YTest = &YTest

}
