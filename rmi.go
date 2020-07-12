package rmi

import (
	"errors"
	"math"
	"math/big"
	"sort"
)

/*
Node in the leanred index tree
m: slope of the current node model
b: intercept of current node model
w: x intercept of the model mw + b = 0
children: array of child nodes (if they exist)
*/
type Node struct {
	m, b, w *big.Float // mx + b and w is the x intercept (mw + b = 0)
}

/*
RMI data structure
Learns the CDF of the data using recursive linear regressions.
root: root node in the model
width: number of data chuncks the data is split into at each layer
depth: recursive depth of the model
nodes: all nodes in the model
*/
type RMI struct {
	width, depth int       // width and depth of the rmi
	root         *Node     // top most node in rmi
	nodes        [][]*Node // each []*Node is all the nodes of a layer
	maxIndex     int       // maximum index in the data structure
}

// NewRMI create a new recursive model index structure with the provided parameters
// see https://dl.acm.org/doi/pdf/10.1145/3183713.3196909?download=true
// for details on the datastructure
func NewRMI(
	values []*big.Int,
	width int,
	depth int) (*RMI, error) {

	// values must be provided in sorted order
	isSorted := sort.SliceIsSorted(values, func(i, j int) bool {
		return values[i].Cmp(values[j]) == -1
	})

	if !isSorted {
		return nil, errors.New("values must be in sorted order")
	}

	indices := make([]*big.Int, len(values))

	// set indices to be the index of each (sorted) value
	for i := range values {
		indices[i] = big.NewInt(int64(i))
	}

	nodes := make([][]*Node, depth)
	layerSize := 1
	for i := range nodes {
		nodes[i] = make([]*Node, layerSize)
		layerSize *= width
	}

	rmi := RMI{}
	rmi.maxIndex = len(values) - 1
	rmi.nodes = nodes
	rmi.width = width
	rmi.depth = depth

	// build the RMI
	rmi.root = rmi.buildRecursive(values, indices, big.NewInt(0), 0, 0)

	return &rmi, nil
}

// GetIndex returns the approximate index for the provided value query
// this is done by having each model (starting from the root) predict
// the model at the subsequent layer that should be queried
func (rmi *RMI) GetIndex(value *big.Int) int {

	width := big.NewFloat(float64(rmi.width))

	// current node that is going to predict the next model for the value
	currentNode := rmi.root

	nextLayer := 1

	// iterate through all nodes
	for {

		m := currentNode.m
		b := currentNode.b
		res := big.NewFloat(0)

		if nextLayer == rmi.depth {
			// reached the leaf layer; return the predicted index (not divided by the width)
			nextIndex64, _ := res.Mul(m, new(big.Float).SetInt(value)).Add(res, b).Int64()
			nextIndex := int(nextIndex64)
			if nextIndex > rmi.maxIndex {
				return rmi.maxIndex
			} else if nextIndex < 0 {
				return 0
			}

			return nextIndex
		}

		// take the model prediction and figure out which child
		// node to select by dividing by layer width
		res.Mul(m, new(big.Float).SetInt(value)).Add(res, b) // mx+b
		res.Quo(res, big.NewFloat(float64(rmi.maxIndex)))    // compute index relative to max index (percentage)
		res.Mul(res, width)                                  // * number of nodes to get index of the responsible node
		nextIndex64, _ := res.Int64()
		nextIndex := int(nextIndex64)

		// make sure the predicted index is within the bounds
		if nextIndex < 0 {
			nextIndex = 0
		} else if nextIndex >= len(rmi.nodes[nextLayer]) {
			nextIndex = len(rmi.nodes[nextLayer]) - 1
		}

		currentNode = rmi.nodes[nextLayer][nextIndex]
		nextLayer++
		width.Mul(width, big.NewFloat(float64(rmi.width)))
	}
}

// Builds the RMI structure recursively from top
// Note: doesnt create new arrays, calculates on same array given two boundaries
func (rmi *RMI) buildRecursive(
	values []*big.Int,
	indices []*big.Int,
	offset *big.Int,
	currentDepth int,
	locationInLayer int) *Node {

	node := &Node{}

	rmi.nodes[currentDepth][locationInLayer] = node

	// compute linear regression for the data of this node
	// m: slope
	// b: constant
	// w: x intercept for the linear regression
	b := big.NewFloat(0.0)
	m := big.NewFloat(0.0)
	w := big.NewFloat(0.0)

	if len(indices) >= 2 {
		b, m, w = coefficients(values, indices)
	} else {
		// this handles the special case where the node contains fewer than 2 points (can't compute regression).
		// The node must still return an index and so it returns offset
		// (the start index of bucket its ancestor is responsible for)
		b = new(big.Float).SetInt(offset)
	}

	node.b = b
	node.m = m
	node.w = w

	// leaf layer not reached yet, recursivley create children for the current node
	if currentDepth != rmi.depth-1 {
		currentDepth++

		// find the range (number of values) that the current layer must learn
		rangeSize := int(float64(len(values)) / float64(rmi.width))

		//left and right bounds index bounds
		leftIndex := 0
		rightIndex := int(math.Max(0, float64(rangeSize)))

		for i := 0; i < rmi.width; i++ {

			// slice of indicies for the children nodes
			subIndices := make([]*big.Int, 0)

			// make sure that the indices are within bounds
			if rightIndex <= 0 {
				rightIndex = 0
				leftIndex = 0
			} else if rightIndex >= len(indices) {
				rightIndex = len(indices) - 1
			}

			// update the offset; used in case the slice is empty
			// to make sure the node returns the right index
			if leftIndex != rightIndex {
				subIndices = indices[leftIndex:rightIndex]
				offset = subIndices[0]
			}

			rmi.buildRecursive(
				values[leftIndex:rightIndex],
				subIndices,
				offset,
				currentDepth,
				locationInLayer*rmi.width+i)

			leftIndex = rightIndex
			rightIndex = int(math.Max(0, math.Min(float64(rightIndex+rangeSize), float64(len(indices)))-1))
		}
	}

	return node
}
