/** THIS IS AN AUTOMATICALLY GENERATED FILE.  DO NOT MODIFY
 * BY HAND!!
 *
 * Generated by lcm-gen 1.5.0
 **/

#ifndef __project_types_image_points_t_hpp__
#define __project_types_image_points_t_hpp__

#include <lcm/lcm_coretypes.h>

#include <vector>

namespace project_types
{

/// For 4 camera setup
class image_points_t
{
    public:
        int32_t    cam0_n;

        int32_t    cam1_n;

        int32_t    cam2_n;

        int32_t    cam3_n;

        /**
         * LCM Type: float[cam0_n][2]
         */
        std::vector< std::vector< float > > cam0_points;

        /**
         * LCM Type: float[cam1_n][2]
         */
        std::vector< std::vector< float > > cam1_points;

        /**
         * LCM Type: float[cam2_n][2]
         */
        std::vector< std::vector< float > > cam2_points;

        /**
         * LCM Type: float[cam3_n][2]
         */
        std::vector< std::vector< float > > cam3_points;

    public:
        /**
         * Encode a message into binary form.
         *
         * @param buf The output buffer.
         * @param offset Encoding starts at thie byte offset into @p buf.
         * @param maxlen Maximum number of bytes to write.  This should generally be
         *  equal to getEncodedSize().
         * @return The number of bytes encoded, or <0 on error.
         */
        inline int encode(void *buf, int offset, int maxlen) const;

        /**
         * Check how many bytes are required to encode this message.
         */
        inline int getEncodedSize() const;

        /**
         * Decode a message from binary form into this instance.
         *
         * @param buf The buffer containing the encoded message.
         * @param offset The byte offset into @p buf where the encoded message starts.
         * @param maxlen The maximum number of bytes to read while decoding.
         * @return The number of bytes decoded, or <0 if an error occured.
         */
        inline int decode(const void *buf, int offset, int maxlen);

        /**
         * Retrieve the 64-bit fingerprint identifying the structure of the message.
         * Note that the fingerprint is the same for all instances of the same
         * message type, and is a fingerprint on the message type definition, not on
         * the message contents.
         */
        inline static int64_t getHash();

        /**
         * Returns "image_points_t"
         */
        inline static const char* getTypeName();

        // LCM support functions. Users should not call these
        inline int _encodeNoHash(void *buf, int offset, int maxlen) const;
        inline int _getEncodedSizeNoHash() const;
        inline int _decodeNoHash(const void *buf, int offset, int maxlen);
        inline static uint64_t _computeHash(const __lcm_hash_ptr *p);
};

int image_points_t::encode(void *buf, int offset, int maxlen) const
{
    int pos = 0, tlen;
    int64_t hash = getHash();

    tlen = __int64_t_encode_array(buf, offset + pos, maxlen - pos, &hash, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = this->_encodeNoHash(buf, offset + pos, maxlen - pos);
    if (tlen < 0) return tlen; else pos += tlen;

    return pos;
}

int image_points_t::decode(const void *buf, int offset, int maxlen)
{
    int pos = 0, thislen;

    int64_t msg_hash;
    thislen = __int64_t_decode_array(buf, offset + pos, maxlen - pos, &msg_hash, 1);
    if (thislen < 0) return thislen; else pos += thislen;
    if (msg_hash != getHash()) return -1;

    thislen = this->_decodeNoHash(buf, offset + pos, maxlen - pos);
    if (thislen < 0) return thislen; else pos += thislen;

    return pos;
}

int image_points_t::getEncodedSize() const
{
    return 8 + _getEncodedSizeNoHash();
}

int64_t image_points_t::getHash()
{
    static int64_t hash = static_cast<int64_t>(_computeHash(NULL));
    return hash;
}

const char* image_points_t::getTypeName()
{
    return "image_points_t";
}

int image_points_t::_encodeNoHash(void *buf, int offset, int maxlen) const
{
    int pos = 0, tlen;

    tlen = __int32_t_encode_array(buf, offset + pos, maxlen - pos, &this->cam0_n, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __int32_t_encode_array(buf, offset + pos, maxlen - pos, &this->cam1_n, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __int32_t_encode_array(buf, offset + pos, maxlen - pos, &this->cam2_n, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __int32_t_encode_array(buf, offset + pos, maxlen - pos, &this->cam3_n, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    for (int a0 = 0; a0 < this->cam0_n; a0++) {
        tlen = __float_encode_array(buf, offset + pos, maxlen - pos, &this->cam0_points[a0][0], 2);
        if(tlen < 0) return tlen; else pos += tlen;
    }

    for (int a0 = 0; a0 < this->cam1_n; a0++) {
        tlen = __float_encode_array(buf, offset + pos, maxlen - pos, &this->cam1_points[a0][0], 2);
        if(tlen < 0) return tlen; else pos += tlen;
    }

    for (int a0 = 0; a0 < this->cam2_n; a0++) {
        tlen = __float_encode_array(buf, offset + pos, maxlen - pos, &this->cam2_points[a0][0], 2);
        if(tlen < 0) return tlen; else pos += tlen;
    }

    for (int a0 = 0; a0 < this->cam3_n; a0++) {
        tlen = __float_encode_array(buf, offset + pos, maxlen - pos, &this->cam3_points[a0][0], 2);
        if(tlen < 0) return tlen; else pos += tlen;
    }

    return pos;
}

int image_points_t::_decodeNoHash(const void *buf, int offset, int maxlen)
{
    int pos = 0, tlen;

    tlen = __int32_t_decode_array(buf, offset + pos, maxlen - pos, &this->cam0_n, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __int32_t_decode_array(buf, offset + pos, maxlen - pos, &this->cam1_n, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __int32_t_decode_array(buf, offset + pos, maxlen - pos, &this->cam2_n, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    tlen = __int32_t_decode_array(buf, offset + pos, maxlen - pos, &this->cam3_n, 1);
    if(tlen < 0) return tlen; else pos += tlen;

    try {
        this->cam0_points.resize(this->cam0_n);
    } catch (...) {
        return -1;
    }
    for (int a0 = 0; a0 < this->cam0_n; a0++) {
        if(2) {
            this->cam0_points[a0].resize(2);
            tlen = __float_decode_array(buf, offset + pos, maxlen - pos, &this->cam0_points[a0][0], 2);
            if(tlen < 0) return tlen; else pos += tlen;
        }
    }

    try {
        this->cam1_points.resize(this->cam1_n);
    } catch (...) {
        return -1;
    }
    for (int a0 = 0; a0 < this->cam1_n; a0++) {
        if(2) {
            this->cam1_points[a0].resize(2);
            tlen = __float_decode_array(buf, offset + pos, maxlen - pos, &this->cam1_points[a0][0], 2);
            if(tlen < 0) return tlen; else pos += tlen;
        }
    }

    try {
        this->cam2_points.resize(this->cam2_n);
    } catch (...) {
        return -1;
    }
    for (int a0 = 0; a0 < this->cam2_n; a0++) {
        if(2) {
            this->cam2_points[a0].resize(2);
            tlen = __float_decode_array(buf, offset + pos, maxlen - pos, &this->cam2_points[a0][0], 2);
            if(tlen < 0) return tlen; else pos += tlen;
        }
    }

    try {
        this->cam3_points.resize(this->cam3_n);
    } catch (...) {
        return -1;
    }
    for (int a0 = 0; a0 < this->cam3_n; a0++) {
        if(2) {
            this->cam3_points[a0].resize(2);
            tlen = __float_decode_array(buf, offset + pos, maxlen - pos, &this->cam3_points[a0][0], 2);
            if(tlen < 0) return tlen; else pos += tlen;
        }
    }

    return pos;
}

int image_points_t::_getEncodedSizeNoHash() const
{
    int enc_size = 0;
    enc_size += __int32_t_encoded_array_size(NULL, 1);
    enc_size += __int32_t_encoded_array_size(NULL, 1);
    enc_size += __int32_t_encoded_array_size(NULL, 1);
    enc_size += __int32_t_encoded_array_size(NULL, 1);
    enc_size += this->cam0_n * __float_encoded_array_size(NULL, 2);
    enc_size += this->cam1_n * __float_encoded_array_size(NULL, 2);
    enc_size += this->cam2_n * __float_encoded_array_size(NULL, 2);
    enc_size += this->cam3_n * __float_encoded_array_size(NULL, 2);
    return enc_size;
}

uint64_t image_points_t::_computeHash(const __lcm_hash_ptr *)
{
    uint64_t hash = 0xa41ee2f8b075801dLL;
    return (hash<<1) + ((hash>>63)&1);
}

}

#endif