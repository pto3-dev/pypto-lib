/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef QWEN_SOFTMAX_PREFETCH_HPP_
#define QWEN_SOFTMAX_PREFETCH_HPP_
#include <pto/pto-inst.hpp>
using namespace pto;
using SoftRow = Tile<TileType::Vec,float,1,1024,BLayout::RowMajor,1,1024>;
using SoftStat = Tile<TileType::Vec,float,8,1,BLayout::ColMajor,1,1>;
using SoftSmall = Tile<TileType::Vec,float,1,8,BLayout::RowMajor,1,8>;
using SoftBF16 = Tile<TileType::Vec,bfloat16_t,1,1024,BLayout::RowMajor,1,1024>;
using SoftGlobal = GlobalTensor<float,Shape<1,1,1,1,1024>,Stride<1024,1024,1024,1024,1>>;
using SoftOut = GlobalTensor<bfloat16_t,Shape<1,1,1,1,1024>,Stride<1024,1024,1024,1024,1>>;
static AICORE __attribute__((always_inline)) inline void SoftScoreLegacy(
    SoftRow& raw,SoftRow& columns,SoftRow& mask,SoftRow& aux,SoftRow& score,float delta) {
    TMULS(raw,raw,0.0625f);TADDS(mask,columns,delta);pipe_barrier(PIPE_V);
    TMULS(mask,mask,-1.0f);pipe_barrier(PIPE_V);TMAXS(mask,mask,0.0f);pipe_barrier(PIPE_V);
    TMINS(mask,mask,1.0f);pipe_barrier(PIPE_V);TMULS(aux,mask,-1.0f);pipe_barrier(PIPE_V);
    TADDS(aux,aux,1.0f);TMUL(score,raw,mask);pipe_barrier(PIPE_V);
    TMULS(aux,aux,-3.4028234663852886e38f);pipe_barrier(PIPE_V);TADD(score,score,aux);pipe_barrier(PIPE_V);
}
template<int MaxSeq,int Rows,int QueryHeads>
static AICORE __attribute__((always_inline)) inline void SoftmaxFastMask(
    __gm__ bfloat16_t* output,__gm__ int32_t* lengths,__gm__ float* columns,
    __gm__ float* input,int core,int cores) {
    SoftRow raw,col,mask,aux,score,maxVector,denVector,exponent,tmp;
    SoftStat maximum,denominator;SoftSmall maxSmall,denSmall;SoftBF16 result;
    TASSIGN(raw,0);TASSIGN(col,4096);TASSIGN(mask,8192);TASSIGN(aux,12288);
    TASSIGN(score,16384);TASSIGN(maxVector,20480);TASSIGN(denVector,24576);
    TASSIGN(exponent,28672);TASSIGN(tmp,32768);TASSIGN(maximum,36864);TASSIGN(denominator,36896);
    TASSIGN(maxSmall,36928);TASSIGN(denSmall,36960);TASSIGN(result,36992);
    SoftGlobal cg(columns);TLOAD(col,cg);
    set_flag(PIPE_MTE2,PIPE_V,EVENT_ID2);wait_flag(PIPE_MTE2,PIPE_V,EVENT_ID2);
    for(int row=core;row<Rows;row+=cores) {
        const int visible=lengths[row/QueryHeads];
        const int active=visible>1?((visible+1023)/1024<MaxSeq/1024?(visible+1023)/1024:MaxSeq/1024):MaxSeq/1024;
        TEXPANDS(maxVector,-3.4028234663852886e38f);pipe_barrier(PIPE_V);
        TASSIGN(raw,0);
        {SoftGlobal firstG(input+row*MaxSeq);TLOAD(raw,firstG);}
        set_flag(PIPE_MTE2,PIPE_V,EVENT_ID0);
        for(int offset=0;offset<active*1024;offset+=1024) {
            const int slot=(offset/1024)&1;
            TASSIGN(raw,slot?40960:0);
            if(offset+1024<active*1024) {
                const int next=1-slot;
                if(offset>=1024) {
                    if(next)wait_flag(PIPE_V,PIPE_MTE2,EVENT_ID1);else wait_flag(PIPE_V,PIPE_MTE2,EVENT_ID0);
                }
                SoftRow nextRaw;TASSIGN(nextRaw,next?40960:0);
                SoftGlobal nextG(input+row*MaxSeq+offset+1024);TLOAD(nextRaw,nextG);
                if(next)set_flag(PIPE_MTE2,PIPE_V,EVENT_ID1);else set_flag(PIPE_MTE2,PIPE_V,EVENT_ID0);
            }
            if(slot)wait_flag(PIPE_MTE2,PIPE_V,EVENT_ID1);else wait_flag(PIPE_MTE2,PIPE_V,EVENT_ID0);
            if(offset+1024<visible) {TMULS(score,raw,0.0625f);pipe_barrier(PIPE_V);}
            else SoftScoreLegacy(raw,col,mask,aux,score,static_cast<float>(offset-visible));
            TMAX(maxVector,maxVector,score);
            if(slot)set_flag(PIPE_V,PIPE_MTE2,EVENT_ID1);else set_flag(PIPE_V,PIPE_MTE2,EVENT_ID0);
        }
        wait_flag(PIPE_V,PIPE_MTE2,EVENT_ID0);
        if(active>1)wait_flag(PIPE_V,PIPE_MTE2,EVENT_ID1);
        pipe_barrier(PIPE_V);TROWMAX(maximum,maxVector,tmp);
        set_flag(PIPE_V,PIPE_S,EVENT_ID0);wait_flag(PIPE_V,PIPE_S,EVENT_ID0);
        const float maxScalar=maximum.GetValue(0);
        TEXPANDS(maxSmall,maxScalar);TEXPANDS(denVector,0.0f);pipe_barrier(PIPE_V);
        TASSIGN(raw,0);
        {SoftGlobal firstG(input+row*MaxSeq);TLOAD(raw,firstG);}
        set_flag(PIPE_MTE2,PIPE_V,EVENT_ID0);
        for(int offset=0;offset<active*1024;offset+=1024) {
            const int slot=(offset/1024)&1;
            TASSIGN(raw,slot?40960:0);
            if(offset+1024<active*1024) {
                const int next=1-slot;
                if(offset>=1024) {
                    if(next)wait_flag(PIPE_V,PIPE_MTE2,EVENT_ID1);else wait_flag(PIPE_V,PIPE_MTE2,EVENT_ID0);
                }
                SoftRow nextRaw;TASSIGN(nextRaw,next?40960:0);
                SoftGlobal nextG(input+row*MaxSeq+offset+1024);TLOAD(nextRaw,nextG);
                if(next)set_flag(PIPE_MTE2,PIPE_V,EVENT_ID1);else set_flag(PIPE_MTE2,PIPE_V,EVENT_ID0);
            }
            if(slot)wait_flag(PIPE_MTE2,PIPE_V,EVENT_ID1);else wait_flag(PIPE_MTE2,PIPE_V,EVENT_ID0);
            if(offset+1024<visible) {TMULS(score,raw,0.0625f);pipe_barrier(PIPE_V);}
            else SoftScoreLegacy(raw,col,mask,aux,score,static_cast<float>(offset-visible));
            TROWEXPANDSUB(exponent,score,maxSmall);pipe_barrier(PIPE_V);
            TEXP(exponent,exponent);pipe_barrier(PIPE_V);TADD(denVector,denVector,exponent);
            if(slot)set_flag(PIPE_V,PIPE_MTE2,EVENT_ID1);else set_flag(PIPE_V,PIPE_MTE2,EVENT_ID0);
        }
        wait_flag(PIPE_V,PIPE_MTE2,EVENT_ID0);
        if(active>1)wait_flag(PIPE_V,PIPE_MTE2,EVENT_ID1);
        pipe_barrier(PIPE_V);TROWSUM(denominator,denVector,tmp);
        set_flag(PIPE_V,PIPE_S,EVENT_ID0);wait_flag(PIPE_V,PIPE_S,EVENT_ID0);
        const float denScalar=denominator.GetValue(0);TEXPANDS(denSmall,denScalar);pipe_barrier(PIPE_V);
        TASSIGN(raw,0);
        {SoftGlobal firstG(input+row*MaxSeq);TLOAD(raw,firstG);}
        set_flag(PIPE_MTE2,PIPE_V,EVENT_ID0);
        for(int offset=0;offset<MaxSeq;offset+=1024) {
            if(visible>1 && offset>=active*1024) {
                TEXPANDS(result,static_cast<bfloat16_t>(0.0f));
                set_flag(PIPE_V,PIPE_MTE3,EVENT_ID0);wait_flag(PIPE_V,PIPE_MTE3,EVENT_ID0);
                SoftOut og(output+row*MaxSeq+offset);TSTORE(og,result);
                set_flag(PIPE_MTE3,PIPE_V,EVENT_ID0);wait_flag(PIPE_MTE3,PIPE_V,EVENT_ID0);
                continue;
            }
            const int slot=(offset/1024)&1;
            TASSIGN(raw,slot?40960:0);
            if(offset+1024<active*1024) {
                const int next=1-slot;
                if(offset>=1024) {
                    if(next)wait_flag(PIPE_V,PIPE_MTE2,EVENT_ID1);else wait_flag(PIPE_V,PIPE_MTE2,EVENT_ID0);
                }
                SoftRow nextRaw;TASSIGN(nextRaw,next?40960:0);
                SoftGlobal nextG(input+row*MaxSeq+offset+1024);TLOAD(nextRaw,nextG);
                if(next)set_flag(PIPE_MTE2,PIPE_V,EVENT_ID1);else set_flag(PIPE_MTE2,PIPE_V,EVENT_ID0);
            }
            if(slot)wait_flag(PIPE_MTE2,PIPE_V,EVENT_ID1);else wait_flag(PIPE_MTE2,PIPE_V,EVENT_ID0);
            if(offset+1024<visible) {TMULS(score,raw,0.0625f);pipe_barrier(PIPE_V);}
            else SoftScoreLegacy(raw,col,mask,aux,score,static_cast<float>(offset-visible));
            TROWEXPANDSUB(exponent,score,maxSmall);pipe_barrier(PIPE_V);TEXP(exponent,exponent);pipe_barrier(PIPE_V);
            TROWEXPANDDIV(exponent,exponent,denSmall);pipe_barrier(PIPE_V);
            TCVT(result,exponent,RoundMode::CAST_RINT);
            set_flag(PIPE_V,PIPE_MTE3,EVENT_ID0);wait_flag(PIPE_V,PIPE_MTE3,EVENT_ID0);
            SoftOut og(output+row*MaxSeq+offset);TSTORE(og,result);
            set_flag(PIPE_MTE3,PIPE_V,EVENT_ID0);wait_flag(PIPE_MTE3,PIPE_V,EVENT_ID0);
            if(slot)set_flag(PIPE_V,PIPE_MTE2,EVENT_ID1);else set_flag(PIPE_V,PIPE_MTE2,EVENT_ID0);
        }
        wait_flag(PIPE_V,PIPE_MTE2,EVENT_ID0);
        if(active>1)wait_flag(PIPE_V,PIPE_MTE2,EVENT_ID1);
    }
    pipe_barrier(PIPE_ALL);
}
#endif
