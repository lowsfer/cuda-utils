#include "ImageReader.h"
#include "cpp_utils.h"
#include <libheif/heif_cxx.h>
#include <libheif/heif.h>
#include <exiv2/exif.hpp>
#include <exiv2/xmp_exiv2.hpp>
#include <turbojpeg.h>
#include <exiv2/exiv2.hpp>
#include <type_traits>
#include <variant>
#include <boost/version.hpp>
#define BOOST_GIL_JPEG_IS_WORKING (BOOST_VERSION >= 108000)
#if BOOST_GIL_JPEG_IS_WORKING
#include <boost/gil/extension/io/jpeg.hpp>
#else
namespace cudapp {
int scanhead (FILE * infile, int * image_width, int * image_height);
}
#endif

namespace cudapp
{
IImageReader::~IImageReader() = default;

void initXmpParserOnce() {
	static std::once_flag initFlag;
	std::call_once(initFlag, [](){
		Exiv2::XmpParser::initialize();
        Exiv2::XmpProperties::registerNs("drong-dji", "drone-dji");
		::atexit(Exiv2::XmpParser::terminate);
	});
}

// handles both heif and avif
class HeifAvifImageReader
{
	using OptDeleter = DeleterWrapper<heif_decoding_options*, &heif_decoding_options_free>;
	static std::unique_ptr<heif_decoding_options, OptDeleter> createHeifDecOpt() {
		return std::unique_ptr<heif_decoding_options, OptDeleter>{heif_decoding_options_alloc()};
	}
public:
	void setFile(const char* filename) {
		const auto ext = getFileExtension(filename);
		ASSERT(ext == ".avif" || ext == ".heif" || ext == ".heic");
		mFile = filename;
		mCtx.read_from_file(filename);
		ASSERT(mCtx.get_number_of_top_level_images() == 1);
		mHandle = mCtx.get_primary_image_handle();
	}

	uint32_t getWidth() const {
		return cast32u(mHandle.get_ispe_width());
	}
	uint32_t getHeight() const {
		return cast32u(mHandle.get_ispe_height());
	}

	Exiv2::XmpData getXmpData() const {
		Exiv2::XmpData dst{};
		const auto xmpIds = mHandle.get_list_of_metadata_block_IDs("XMP");
		if (xmpIds.empty()) {
			return dst;
		}
		ASSERT(xmpIds.size() == 1);
		const auto id = xmpIds.at(0);
		const auto metadata = mHandle.get_metadata(id);
		const std::string xmpPacket(reinterpret_cast<const char*>(metadata.data()), metadata.size());
		initXmpParserOnce();
		ASSERT(Exiv2::XmpParser::decode(dst, xmpPacket) == 0);
		return dst;
	}
	Exiv2::ExifData getExifData() const {
		Exiv2::ExifData dst{};
		const auto exifIds = mHandle.get_list_of_metadata_block_IDs("Exif");
		if (exifIds.empty()) {
			return dst;
		}
		ASSERT(exifIds.size() == 1);
		const auto id = exifIds.at(0);
		const auto metadata = mHandle.get_metadata(id);
		const auto byteOrder = Exiv2::ExifParser::decode(dst, metadata.data(), metadata.size());
		unused(byteOrder);
		return dst;
	}

	Image8UC3 decodeTo8UC3() const {
		auto opt = createHeifDecOpt();
		opt->ignore_transformations = true;
		opt->convert_hdr_to_8bit = true;
		heif_image* pImg = nullptr;
		heif_decode_image(mHandle.get_raw_image_handle(), &pImg, heif_colorspace::heif_colorspace_RGB, heif_chroma::heif_chroma_interleaved_RGB, opt.get());
		heif::Image img{pImg};
		const auto channel = heif_channel::heif_channel_interleaved;
		ASSERT(img.get_bits_per_pixel(channel) == sizeof(Image8UC3::Pixel) * 8);
		ASSERT(cast32u(img.get_width(channel)) == getWidth());
		ASSERT(cast32u(img.get_height(channel)) == getHeight());
		int stride = 0;
		auto p = img.get_plane(channel, &stride);
		const size_t rowSize = sizeof(Image8UC3::Pixel) * getWidth();
		Image8UC3 result{getWidth(), getHeight()};
		uint8_t* dst = reinterpret_cast<uint8_t*>(result.data());
		for (uint32_t i = 0; i < getHeight(); i++) {
			std::copy_n(p, rowSize, dst);
			p += stride;
			dst += rowSize;
		}
		return result;
	}
private:
	std::string mFile; // don't use fs::path, causing problems on windows
	heif::Context mCtx;
	heif::ImageHandle mHandle;
};

class JpegImageReader
{
public:
	void setFile(const char* filename) {
		const auto ext = getFileExtension(filename);
		ASSERT(ext == ".jpg" || ext == ".jpeg");
		mFile = filename;
		mShape = std::nullopt;
		mMetaData = std::nullopt;
	}

	uint32_t getWidth() const {
		loadShape();
		return mShape.value().width;
	}
	uint32_t getHeight() const {
		loadShape();
		return mShape.value().height;
	}

	Exiv2::XmpData getXmpData() const {
		loadMetaData();
		return mMetaData.value().second;
	}
	Exiv2::ExifData getExifData() const {
		loadMetaData();
		return mMetaData.value().first;
	}

	Image8UC3 decodeTo8UC3() const {
        auto jpegData = loadBinaryFile<uint8_t>(mFile.c_str());
		using TjDeleter = DeleterWrapper<tjhandle, &tjHandleDel>;
		const std::unique_ptr<void, TjDeleter> tj {tjInitDecompress()};
		int width, height;
		ASSERT(tjDecompressHeader(tj.get(), jpegData.data(), jpegData.size(), &width, &height) == 0);
		ASSERT(cast32u(width) == getWidth());
		ASSERT(cast32u(height) == getHeight());
		Image8UC3 img(width, height);
		ASSERT(0 == tjDecompress2(tj.get(), jpegData.data(), jpegData.size(), reinterpret_cast<uint8_t*>(img.data()), width, sizeof(Image8UC3::Pixel) * width, height, TJPF_RGB, TJFLAG_ACCURATEDCT));
		return img;
	}
private:
	static void tjHandleDel(tjhandle h) {
		HOPE(tjDestroy(h) == 0);
	}
	void loadShape() const {
		if (mShape.has_value()){
			return;
		}
		auto& shape = const_cast<std::optional<Shape2D>&>(mShape);
#if BOOST_GIL_JPEG_IS_WORKING
		namespace gil = boost::gil;
		const auto backend = gil::read_image_info(mFile.c_str(), gil::jpeg_tag());
		shape = Shape2D{.width = backend._info._width, .height = backend._info._height};
#else
		auto const file = fopen(mFile.c_str(), "rb");
		int width = 0;
		int height = 0;
		ASSERT(scanhead(file, &width, &height) != 0);
		shape = Shape2D{.width = cast32u(width), .height = cast32u(height)};
#endif
	}
	void loadMetaData() const {
		if (mMetaData.has_value()) {
			return;
		}
		initXmpParserOnce();
		auto img = Exiv2::ImageFactory::open(mFile.c_str());
		img->readMetadata();
		auto& metadata = const_cast<std::optional<std::pair<Exiv2::ExifData, Exiv2::XmpData>>&>(mMetaData);
		metadata = std::make_pair(img->exifData(), img->xmpData());
	}

private:
	std::string mFile; // don't use fs::path, causing problems on windows
	std::optional<Shape2D> mShape;
	std::optional<std::pair<Exiv2::ExifData, Exiv2::XmpData>> mMetaData;
};

class ImageReader : public IImageReader
{
public:
	void setFile(const char* filename) override {
		const auto ext = getFileExtension(filename);
		if (ext == ".jpg" || ext == ".jpeg") {
			setFile(filename, ImageFileType::kJPEG);
		}
		else if (ext == ".avif") {
			setFile(filename, ImageFileType::kAVIF);
		}
		else if (ext == ".heif" || ext == ".heic") {
			setFile(filename, ImageFileType::kHEIF);
		}
		else {
			throw std::runtime_error("unknown image file type");
		}
	}
	void setFile(const char* filename, ImageFileType type) override {
		mCurrentType = type;
		std::visit([filename](auto* p){p->setFile(filename);}, getReader());
	}

	// this ignores transforms
	uint32_t getWidth() const override {
		return std::visit([this](auto* p){return p->getWidth();}, getReader());
	}
	uint32_t getHeight() const override {
		return std::visit([this](auto* p){return p->getHeight();}, getReader());
	}

	Exiv2::XmpData getXmpData() const {
		return std::visit([this](auto* p){return p->getXmpData();}, getReader());
	}
	Exiv2::ExifData getExifData() const {
		return std::visit([this](auto* p){return p->getExifData();}, getReader());
	}

	Image8UC3 decodeTo8UC3() const {
		return std::visit([this](auto* p){return p->decodeTo8UC3();}, getReader());
	}
private:
	std::tuple<std::unique_ptr<JpegImageReader>, std::unique_ptr<HeifAvifImageReader>> mReaders;
	ImageFileType mCurrentType;

private:
	std::variant<JpegImageReader*, HeifAvifImageReader*> getReader() {
		switch (mCurrentType) {
		case ImageFileType::kJPEG: {
				auto& opt = std::get<0>(mReaders);
				if (!opt) {
					opt = std::make_unique<JpegImageReader>();
				}
				return opt.get();
			}
		case ImageFileType::kAVIF:
		case ImageFileType::kHEIF: {
				auto& opt = std::get<1>(mReaders);
				if (!opt) {
					opt = std::make_unique<HeifAvifImageReader>();
				}
				return opt.get();
			}
		}
		DIE("You should never reach here\n");
	}
	std::variant<const JpegImageReader*, const HeifAvifImageReader*> getReader() const {
		auto tmp = const_cast<ImageReader*>(this)->getReader();
		std::variant<const JpegImageReader*, const HeifAvifImageReader*> dst;
		std::visit([&dst](auto p){dst = p;}, tmp);
		return dst;
	}
};

std::unique_ptr<IImageReader> createImageReader() {
	return std::make_unique<ImageReader>();
}

Shape2D getImageSize(const char* filename) {
	const auto ext = getFileExtension(filename);
	if (ext == ".jpg" || ext == ".jpeg") {
		JpegImageReader reader{};
		reader.setFile(filename);
		return Shape2D{.width = reader.getWidth(), .height = reader.getHeight()};
	}
	else if (ext == ".avif" || ext == ".heif" || ext == ".heic") {
		HeifAvifImageReader reader{};
		reader.setFile(filename);
		return Shape2D{.width = reader.getWidth(), .height = reader.getHeight()};
	}
	else {
		throw std::runtime_error("unknown image file type");
	}
}

Image8U rgbToGray(const Image8UC3& src) {
	Image8U dst{src.width(), src.height()};
	std::transform(src.begin(), src.end(), dst.begin(), [](const Image8UC3::Pixel& p){
		return uint8_t(clamp(std::round(0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]), 0.f, 255.f));
	});
	return dst;
}

// from http://carnage-melon.tom7.org/stuff/jpegsize.html
// The author claims: This routine is as public domain as is legally allowed.

/* portions derived from IJG code */

#define readbyte(a,b) do if(((a)=getc((b))) == EOF) return 0; while (0)
#define readword(a,b) do { int cc_=0,dd_=0; \
                          if((cc_=getc((b))) == EOF \
        		  || (dd_=getc((b))) == EOF) return 0; \
                          (a) = (cc_<<8) + (dd_); \
                          } while(0)


int scanhead (FILE * infile, int * image_width, int * image_height) {
  int marker=0;
  int dummy=0;
  if ( getc(infile) != 0xFF || getc(infile) != 0xD8 )
    return 0;

  for (;
      ;) {


    int discarded_bytes=0;
    readbyte(marker,infile);
    while (marker != 0xFF) {
      discarded_bytes++;
      readbyte(marker,infile);
    }
    do readbyte(marker,infile); while (marker == 0xFF);

    if (discarded_bytes != 0) return 0;
   
    switch (marker) {
    case 0xC0:
    case 0xC1:
    case 0xC2:
    case 0xC3:
    case 0xC5:
    case 0xC6:
    case 0xC7:
    case 0xC9:
    case 0xCA:
    case 0xCB:
    case 0xCD:
    case 0xCE:
    case 0xCF: {
      readword(dummy,infile);	/* usual parameter length count */
      readbyte(dummy,infile);
      readword((*image_height),infile);
      readword((*image_width),infile);
      readbyte(dummy,infile);

      return 1;
      break;
      }
    case 0xDA:
    case 0xD9:
      return 0;
    default: {
	int length;
	
	readword(length,infile);

	if (length < 2)
	  return 0;
	length -= 2;
	while (length > 0) {
	  readbyte(dummy, infile);
	  length--;
	}
      }
      break;
    }
  }
}


} // namespace cudapp
