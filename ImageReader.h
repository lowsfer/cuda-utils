#pragma once
#include <cstdint>
#include <vector>
#include <array>
#include "macros.h"
#include <memory>

namespace Exiv2
{
class XmpData;
class ExifData;
}

namespace cudapp
{
template <typename T = std::array<uint8_t, 3>>
class ImageT
{
public:
	using Pixel = T;
	ImageT(std::vector<T> data, uint32_t width_, uint32_t height_) : mWidth{width_}, mHeight{height_}, mData{std::move(data)}{
		ASSERT(width() * height() == mData.size());
	}
	ImageT(uint32_t width_, uint32_t height_) : mWidth{width_}, mHeight{height_}{
		mData.resize(width() * height());
	}
	bool empty() const {
		assert((width() == 0 || height() == 0) == mData.empty());
		return mData.empty();
	}
	const T* data() const {return mData.data();}
	T* data() {return mData.data();}
	uint32_t width() const {return mWidth;}
	uint32_t height() const {return mHeight;}
	const T& operator()(uint32_t h, uint32_t w) const {return mData[width() * h + w];}
	T& operator()(uint32_t h, uint32_t w) {return mData[width() * h + w];}
	
	Pixel* begin() {return &mData[0];}
	Pixel* end() {return begin() + mData.size();}
	const Pixel* begin() const {return &mData[0];}
	const Pixel* end() const {return begin() + mData.size();}
private:
	uint32_t mWidth;
	uint32_t mHeight;
	std::vector<T> mData;
};

struct Shape2D
{
    uint32_t width;
    uint32_t height;
};

enum ImageFileType : int8_t
{
	kJPEG,
	kAVIF,
	kHEIF
	//@fixme: Add HEIC and JXL in the future
};

Shape2D getImageSize(const char* filename);

using Image8UC3 = ImageT<std::array<uint8_t, 3>>;
using Image8U = ImageT<uint8_t>;

Image8U rgbToGray(const Image8UC3& src);

// This image reader ignores transforms for getWidth()/getHeight()/decode*()
class IImageReader
{
public:
	virtual ~IImageReader();

	virtual void setFile(const char* filename) = 0;
	virtual void setFile(const char* filename, ImageFileType type) = 0;

	// this ignores transforms
	virtual uint32_t getWidth() const = 0;
	virtual uint32_t getHeight() const = 0;

	virtual Exiv2::XmpData getXmpData() const = 0;
	virtual Exiv2::ExifData getExifData() const = 0;

	virtual Image8UC3 decodeTo8UC3() const = 0;
};
void initXmpParserOnce();
std::unique_ptr<IImageReader> createImageReader();

}

