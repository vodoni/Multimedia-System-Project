PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/opt/working/opencv/release/lib/pkgconfig export PKG_CONFIG_PATH
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

rm -rf io/*
rm -rf output/*/*
g++ proj.cpp -lpthread -std=c++11 `pkg-config opencv --cflags --libs` -o main.o

