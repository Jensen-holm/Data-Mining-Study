package nn

import (
	"math"
	"math/rand"

	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/gonum/mat"
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

	// to make linear algebra easier & faster,
	// we convert these dataframes that we are
	// performing potentially expensive computations
	// on into gonum matrices since we no longer need the
	// column names.
	nn.XTrain = df2mat(&XTrain)
	nn.YTrain = df2mat(&YTrain)
	nn.XTest = df2mat(&XTest)
	nn.YTest = df2mat(&YTest)
}

// df2mat -> converts gota dataframe into gonum matrix
func df2mat(df *dataframe.DataFrame) *mat.Dense {
	m := mat.NewDense(df.Nrow(), df.Ncol(), nil)
	for i := 0; i < df.Nrow(); i++ {
		for j := 0; j < df.Ncol(); j++ {
			m.Set(i, j, df.Elem(i, j).Float())
		}
	}
	return m
}
