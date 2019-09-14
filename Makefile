BINARY="checker"
DATASET_DIR="data"
MODELS_DIR="model"
RESULT_DIR="result"

build:
	GO111MODULE=off go build -o ${BINARY} .

clean:
	rm ${BINARY} && rm -rf ${RESULT_DIR}/*

run:
	./${BINARY} -dataset ${DATASET_DIR} -model ${MODELS_DIR} -output ${RESULT_DIR}

# GO111MODULE=off go build -o ${BINARY} && clear && nice --20 ${BINARY} -dataset ${DATASET_DIR} -model ${MODELS_DIR} -output ${RESULT_DIR}
