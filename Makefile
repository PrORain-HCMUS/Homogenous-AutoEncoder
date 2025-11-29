CXX = g++
CXXFLAGS = -O2 -std=c++17
INCLUDES = -Iinclude

SRC = src/main.cpp \
      src/dataset.cpp \
      src/layers_cpu.cpp \
      src/autoencoder.cpp

TARGET = cpu_train

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(SRC)

clean:
	rm -f $(TARGET)
