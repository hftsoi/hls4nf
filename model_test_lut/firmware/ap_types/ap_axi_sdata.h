// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
/*****************************************************************************
 *
 *     Author: Xilinx, Inc.
 *
 *     This text contains proprietary, confidential information of
 *     Xilinx, Inc. , is distributed by under license from Xilinx,
 *     Inc., and may be used, copied and/or disclosed only pursuant to
 *     the terms of a valid license agreement with Xilinx, Inc.
 *
 *     XILINX IS PROVIDING THIS DESIGN, CODE, OR INFORMATION "AS IS"
 *     AS A COURTESY TO YOU, SOLELY FOR USE IN DEVELOPING PROGRAMS AND
 *     SOLUTIONS FOR XILINX DEVICES.  BY PROVIDING THIS DESIGN, CODE,
 *     OR INFORMATION AS ONE POSSIBLE IMPLEMENTATION OF THIS FEATURE,
 *     APPLICATION OR STANDARD, XILINX IS MAKING NO REPRESENTATION
 *     THAT THIS IMPLEMENTATION IS FREE FROM ANY CLAIMS OF INFRINGEMENT,
 *     AND YOU ARE RESPONSIBLE FOR OBTAINING ANY RIGHTS YOU MAY REQUIRE
 *     FOR YOUR IMPLEMENTATION.  XILINX EXPRESSLY DISCLAIMS ANY
 *     WARRANTY WHATSOEVER WITH RESPECT TO THE ADEQUACY OF THE
 *     IMPLEMENTATION, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OR
 *     REPRESENTATIONS THAT THIS IMPLEMENTATION IS FREE FROM CLAIMS OF
 *     INFRINGEMENT, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *     FOR A PARTICULAR PURPOSE.
 *
 *     Xilinx products are not intended for use in life support appliances,
 *     devices, or systems. Use in such applications is expressly prohibited.
 *
 *     __VIVADO_HLS_COPYRIGHT-INFO__
 *
 *****************************************************************************/

/*
 * This file contains the definition of the data types for AXI streaming. 
 * ap_axi_s is a signed interpretation of the AXI stream
 * ap_axi_u is an unsigned interpretation of the AXI stream
 */

#ifndef __AP__AXI_SDATA__
#define __AP__AXI_SDATA__

#include <climits>
#include "ap_int.h"
//#include "ap_fixed.h"
template <int _AP_W, int _AP_I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int _AP_N>
struct ap_fixed;
template <int _AP_W, int _AP_I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int _AP_N>
struct ap_ufixed;

namespace hls {

template <typename T> constexpr std::size_t bitwidth = sizeof(T) * CHAR_BIT;

template <std::size_t W> constexpr std::size_t bitwidth<ap_int<W>> = W;
template <std::size_t W> constexpr std::size_t bitwidth<ap_uint<W>> = W;
template <int _AP_W, int _AP_I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int _AP_N>
constexpr std::size_t bitwidth<ap_fixed<_AP_W, _AP_I, _AP_Q, _AP_O, _AP_N>> = _AP_W;
template <int _AP_W, int _AP_I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int _AP_N>
constexpr std::size_t bitwidth<ap_ufixed<_AP_W, _AP_I, _AP_Q, _AP_O, _AP_N>> = _AP_W;

template <typename T>
constexpr std::size_t bytewidth = (bitwidth<T> + CHAR_BIT - 1) / CHAR_BIT;

template <typename T, std::size_t WUser, std::size_t WId, std::size_t WDest> struct axis {
  static constexpr std::size_t NewWUser = (WUser == 0) ? 1 : WUser;
  static constexpr std::size_t NewWId = (WId == 0) ? 1 : WId;
  static constexpr std::size_t NewWDest = (WDest == 0) ? 1 : WDest;
  T data;
  ap_uint<bytewidth<T>> keep;
  ap_uint<bytewidth<T>> strb;
  ap_uint<NewWUser> user;
  ap_uint<1> last;
  ap_uint<NewWId> id;
  ap_uint<NewWDest> dest;

  NODEBUG ap_uint<NewWUser> *get_user_ptr() { 
#pragma HLS inline
    return (WUser == 0) ? nullptr : &user;
  }
  NODEBUG ap_uint<NewWId> *get_id_ptr() {
#pragma HLS inline
    return (WId == 0) ? nullptr : &id;
  }
  NODEBUG ap_uint<NewWDest> *get_dest_ptr() {
#pragma HLS inline
    return (WDest == 0) ? nullptr : &dest;
  }
};

} // namespace hls

template <std::size_t WData, std::size_t WUser, std::size_t WId, std::size_t WDest>
using ap_axis = hls::axis<ap_int<WData>, WUser, WId, WDest>;

template <std::size_t WData, std::size_t WUser, std::size_t WId, std::size_t WDest>
using ap_axiu = hls::axis<ap_uint<WData>, WUser, WId, WDest>;

// Isolate out qdma_axis from hls::axis for special APIs.
template <std::size_t WData, std::size_t WUser, std::size_t WId, std::size_t WDest>
struct qdma_axis;

template <std::size_t WData> struct qdma_axis<WData, 0, 0, 0> {
  //  private:
  static constexpr std::size_t kBytes = (WData + 7) / 8;

  ap_uint<WData> data;
  ap_uint<kBytes> keep;
  ap_uint<1> strb;
  ap_uint<1> user;
  ap_uint<1> last;
  ap_uint<1> id;
  ap_uint<1> dest;

  NODEBUG ap_uint<1> *get_strb_ptr() {
#pragma HLS inline
    return nullptr;
  }
  NODEBUG ap_uint<1> *get_user_ptr() {
#pragma HLS inline
    return nullptr;
  }
  NODEBUG ap_uint<1> *get_id_ptr() {
#pragma HLS inline
    return nullptr;
  }
  NODEBUG ap_uint<1> *get_dest_ptr() {
#pragma HLS inline
    return nullptr;
  }

  //  public:
  NODEBUG ap_uint<WData> get_data() const {
#pragma HLS inline
    return data;
  }
  NODEBUG ap_uint<kBytes> get_keep() const {
#pragma HLS inline
    return keep;
  }
  NODEBUG ap_uint<1> get_last() const {
#pragma HLS inline
    return last;
  }

  NODEBUG void set_data(const ap_uint<WData> &d) {
#pragma HLS inline
    data = d;
  }
  NODEBUG void set_keep(const ap_uint<kBytes> &k) {
#pragma HLS inline
    keep = k;
  }
  NODEBUG void set_last(const ap_uint<1> &l) {
#pragma HLS inline
    last = l;
  }
  NODEBUG void keep_all() {
#pragma HLS inline
    ap_uint<kBytes> k = 0;
    keep = ~k;
  }

  NODEBUG qdma_axis() {
#pragma HLS inline
    ;
  }
  NODEBUG qdma_axis(ap_uint<WData> d) : data(d) {
#pragma HLS inline
    ;
  }
  NODEBUG qdma_axis(ap_uint<WData> d, ap_uint<kBytes> k) : data(d), keep(k) {
#pragma HLS inline
    ;
  }
  NODEBUG qdma_axis(ap_uint<WData> d, ap_uint<kBytes> k, ap_uint<1> l)
      : data(d), keep(k), last(l) {
#pragma HLS inline
    ;
  }
  NODEBUG qdma_axis(const qdma_axis<WData, 0, 0, 0> &d)
      : data(d.data), keep(d.keep), last(d.last) {
#pragma HLS inline
    ;
  }
  NODEBUG qdma_axis &operator=(const qdma_axis<WData, 0, 0, 0> &d) {
#pragma HLS inline
    data = d.data;
    keep = d.keep;
    last = d.last;
    return *this;
  }
};

#ifdef AESL_SYN 
#if ((__clang_major__ != 3) || (__clang_minor__ != 1))
#include "hls_stream.h"
namespace hls {

template <typename T, std::size_t WUser, std::size_t WId, std::size_t WDest>
class stream<axis<T, WUser, WId, WDest>> final {
  typedef axis<T, WUser, WId, WDest> __STREAM_T__;

public:
  /// Constructors
  INLINE NODEBUG stream() {}

  INLINE NODEBUG stream(const char *name) { (void)name; }

  /// Make copy constructor and assignment operator private
private:
  INLINE NODEBUG stream(const stream<__STREAM_T__> &chn) : V(chn.V) {}

public:
  /// Overload >> and << operators to implement read() and write()
  INLINE NODEBUG void operator>>(__STREAM_T__ &rdata) { read(rdata); }

  INLINE NODEBUG void operator<<(const __STREAM_T__ &wdata) { write(wdata); }

  /// empty & full
  NODEBUG bool empty() {
#pragma HLS inline
    bool tmp = __fpga_axis_valid(&V.data, &V.keep, &V.strb, V.get_user_ptr(),
                                 &V.last, V.get_id_ptr(), V.get_dest_ptr());
    return !tmp;
  }

  NODEBUG bool full() {
#pragma HLS inline
    bool tmp = __fpga_axis_ready(&V.data, &V.keep, &V.strb, V.get_user_ptr(),
                                 &V.last, V.get_id_ptr(), V.get_dest_ptr());
    return !tmp;
  }

  /// Blocking read
  NODEBUG void read(__STREAM_T__ &dout) {
#pragma HLS inline
    __STREAM_T__ tmp;
    __fpga_axis_pop(&V.data, &V.keep, &V.strb, V.get_user_ptr(), &V.last,
                    V.get_id_ptr(), V.get_dest_ptr(), &tmp.data, &tmp.keep,
                    &tmp.strb, tmp.get_user_ptr(), &tmp.last, tmp.get_id_ptr(),
                    tmp.get_dest_ptr());
    dout = tmp;
  }

  NODEBUG __STREAM_T__ read() {
#pragma HLS inline
    __STREAM_T__ tmp;
    __fpga_axis_pop(&V.data, &V.keep, &V.strb, V.get_user_ptr(), &V.last,
                    V.get_id_ptr(), V.get_dest_ptr(), &tmp.data, &tmp.keep,
                    &tmp.strb, tmp.get_user_ptr(), &tmp.last, tmp.get_id_ptr(),
                    tmp.get_dest_ptr());
    return tmp;
  }

  /// Blocking write
  NODEBUG void write(const __STREAM_T__ &din) {
#pragma HLS inline
    __STREAM_T__ tmp = din;
    __fpga_axis_push(&V.data, &V.keep, &V.strb, V.get_user_ptr(), &V.last,
                     V.get_id_ptr(), V.get_dest_ptr(), &tmp.data, &tmp.keep,
                     &tmp.strb, tmp.get_user_ptr(), &tmp.last, tmp.get_id_ptr(),
                     tmp.get_dest_ptr());
  }

  /// Non-Blocking read
  NODEBUG bool read_nb(__STREAM_T__ &dout) {
#pragma HLS inline
    __STREAM_T__ tmp;
    if (__fpga_axis_nb_pop(&V.data, &V.keep, &V.strb, V.get_user_ptr(), &V.last,
                           V.get_id_ptr(), V.get_dest_ptr(), &tmp.data,
                           &tmp.keep, &tmp.strb, tmp.get_user_ptr(),
                           &tmp.last, tmp.get_id_ptr(), tmp.get_dest_ptr())) {
      dout = tmp;
      return true;
    } else {
      return false;
    }
  }

  /// Non-Blocking write
  NODEBUG bool write_nb(const __STREAM_T__ &in) {
#pragma HLS inline
    __STREAM_T__ tmp = in;
    bool full_n = __fpga_axis_nb_push(
        &V.data, &V.keep, &V.strb, V.get_user_ptr(), &V.last, V.get_id_ptr(),
        V.get_dest_ptr(), &tmp.data, &tmp.keep, &tmp.strb, tmp.get_user_ptr(),
        &tmp.last, tmp.get_id_ptr(), tmp.get_dest_ptr());
    return full_n;
  }

private:
  __STREAM_T__ V NO_CTOR;
};

// specialization for qdma
template <std::size_t WData>
class stream<qdma_axis<WData, 0, 0, 0>> {
  typedef qdma_axis<WData, 0, 0, 0> __STREAM_T__;

public:
  /// Constructors
  INLINE NODEBUG stream() {}

  INLINE NODEBUG stream(const char *name) { (void)name; }

  /// Make copy constructor and assignment operator private
private:
  INLINE NODEBUG stream(const stream<__STREAM_T__> &chn) : V(chn.V) {}

public:
  /// Overload >> and << operators to implement read() and write()
  INLINE NODEBUG void operator>>(__STREAM_T__ &rdata) { read(rdata); }

  INLINE NODEBUG void operator<<(const __STREAM_T__ &wdata) { write(wdata); }

  /// empty & full
  NODEBUG bool empty() {
#pragma HLS inline
    bool tmp = __fpga_axis_valid(&V.data, &V.keep, V.get_strb_ptr(), V.get_user_ptr(),
                                 &V.last, V.get_id_ptr(), V.get_dest_ptr());
    return !tmp;
  }

  NODEBUG bool full() {
#pragma HLS inline
    bool tmp = __fpga_axis_ready(&V.data, &V.keep, V.get_strb_ptr(), V.get_user_ptr(),
                                 &V.last, V.get_id_ptr(), V.get_dest_ptr());
    return !tmp;
  }

  /// Blocking read
  NODEBUG void read(__STREAM_T__ &dout) {
#pragma HLS inline
    __STREAM_T__ tmp;
    __fpga_axis_pop(&V.data, &V.keep, V.get_strb_ptr(), V.get_user_ptr(),
                    &V.last, V.get_id_ptr(), V.get_dest_ptr(), &tmp.data,
                    &tmp.keep, tmp.get_strb_ptr(), tmp.get_user_ptr(),
                    &tmp.last, tmp.get_id_ptr(), tmp.get_dest_ptr());
    dout = tmp;
  }

  NODEBUG __STREAM_T__ read() {
#pragma HLS inline
    __STREAM_T__ tmp;
    __fpga_axis_pop(&V.data, &V.keep, V.get_strb_ptr(), V.get_user_ptr(), &V.last,
                    V.get_id_ptr(), V.get_dest_ptr(), &tmp.data, &tmp.keep,
                    tmp.get_strb_ptr(), tmp.get_user_ptr(), &tmp.last, tmp.get_id_ptr(),
                    tmp.get_dest_ptr());
    return tmp;
  }

  /// Blocking write
  NODEBUG void write(const __STREAM_T__ &din) {
#pragma HLS inline
    __STREAM_T__ tmp = din;
    __fpga_axis_push(&V.data, &V.keep, V.get_strb_ptr(), V.get_user_ptr(), &V.last,
                     V.get_id_ptr(), V.get_dest_ptr(), &tmp.data, &tmp.keep,
                     tmp.get_strb_ptr(), tmp.get_user_ptr(), &tmp.last, tmp.get_id_ptr(),
                     tmp.get_dest_ptr());
  }

  /// Non-Blocking read
  NODEBUG bool read_nb(__STREAM_T__ &dout) {
#pragma HLS inline
    __STREAM_T__ tmp;

    if (__fpga_axis_nb_pop(&V.data, &V.keep, &V.strb, V.get_user_ptr(), &V.last,
                           V.get_id_ptr(), V.get_dest_ptr(), &tmp.data,
                           &tmp.keep, &tmp.strb, tmp.get_user_ptr(),
                           &tmp.last, tmp.get_id_ptr(), tmp.get_dest_ptr())) {
      dout = tmp;
      return true;
    } else {
      return false;
    }
  }

  /// Non-Blocking write
  NODEBUG bool write_nb(const __STREAM_T__ &in) {
#pragma HLS inline
    __STREAM_T__ tmp = in;
    bool full_n = __fpga_axis_nb_push(
        &V.data, &V.keep, V.get_strb_ptr(), V.get_user_ptr(), &V.last, V.get_id_ptr(),
        V.get_dest_ptr(), &tmp.data, &tmp.keep, tmp.get_strb_ptr(), tmp.get_user_ptr(),
        &tmp.last, tmp.get_id_ptr(), tmp.get_dest_ptr());
    return full_n;
  }

private:
  __STREAM_T__ V NO_CTOR;
};

} // namespace hls
#endif
#endif
#endif
