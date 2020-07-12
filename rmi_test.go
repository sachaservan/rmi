package rmi

import (
	"math"
	"math/big"
	"math/rand"
	"sort"
	"testing"
	"time"
)

// test configuration parameters
const RMIWidthParameter int = 10
const RMIDepthParameter int = 2
const MinDataValue int = 0
const MaxDataValue int = math.MaxInt64
const NumDataPoints int = 10000
const NumQueries int = 20

// error tolerance (true index - pred index); this is a heuristic that
// depends on the data parameters above
// TODO: devise a more rigorous notion of accuracy and measurment for this test
const QueryAccuracyThreshold float64 = 200.0

// generates 'n' random values in the range min..max
func generateRandomData(n int, min int, max int) []*big.Int {
	values := make([]*big.Int, n)
	for i := range values {
		values[i] = big.NewInt(int64(rand.Intn(max-min) + min))
	}

	return values
}

// Generates random data and constructs an RMI
// data structure over it.
// All parameters are specified up top.
func generateTestRMI() (*RMI, []*big.Int, error) {

	values := generateRandomData(NumDataPoints, MinDataValue, MaxDataValue)

	// sort the values in increasing order
	sort.Slice(values, func(i, j int) bool {
		return values[i].Cmp(values[j]) == -1
	})

	// build the rmi over the sorted values
	rmi, err := NewRMI(values, RMIWidthParameter, RMIDepthParameter)

	return rmi, values, err
}

func TestBuild(t *testing.T) {
	rand.Seed(time.Now().Unix())

	_, _, err := generateTestRMI()

	if err != nil {
		t.Fatalf("Failed to build RMI %v\n", err)
	}
}

func distanceToValueFromIndex(values []*big.Int, value *big.Int, index int) int {

	distanceRight := 0
	for i := index; i < len(values); i++ {
		cmpRes := values[i].Cmp(value)
		if cmpRes == 0 {
			break
		} else if cmpRes == 1 {
			// value is not on the right side
			distanceRight = math.MaxInt32
		}

		distanceRight++
	}

	distanceLeft := 0
	for i := index; i >= 0; i-- {
		cmpRes := values[i].Cmp(value)
		if cmpRes == 0 {
			break
		} else if cmpRes == -1 {
			// value is not on the left side
			distanceLeft = math.MaxInt32
		}

		distanceLeft++
	}

	return int(math.Min(float64(distanceLeft), float64(distanceRight)))
}

/////////////////////////////////////////////////////////////////
// TESTS
/////////////////////////////////////////////////////////////////

// executes a query over the RMI data structure.
// run with 'go test -v -run TestGetIndex' to see log outputs.
func TestGetIndex(t *testing.T) {
	rand.Seed(time.Now().Unix())

	rmi, values, _ := generateTestRMI()

	avgErr := 0.0
	for i := 0; i < NumQueries; i++ {
		actualIndex := rand.Intn(NumDataPoints)
		predictedIndex := rmi.GetIndex(values[actualIndex])

		err := float64(distanceToValueFromIndex(values, values[actualIndex], predictedIndex))
		avgErr += err
		t.Logf("dist err %v (actual index = %v)\n", err, actualIndex)

		// TODO: have error guarantees on the model and check them here; 100 is arbitrary
		if err > QueryAccuracyThreshold {
			t.Fatalf(
				"Error is too large: %v > %v",
				err,
				QueryAccuracyThreshold,
			)
		}
	}

	t.Logf("avgErr = %v \n", avgErr/float64(len(values)))
}

func BenchmarkBuild(b *testing.B) {

	values := generateRandomData(NumDataPoints, MinDataValue, MaxDataValue)

	// sort the values in increasing order
	sort.Slice(values, func(i, j int) bool {
		return values[i].Cmp(values[j]) == -1
	})

	// benchmark index build time
	for i := 0; i < b.N; i++ {
		NewRMI(values, RMIWidthParameter, RMIDepthParameter)
	}
}
