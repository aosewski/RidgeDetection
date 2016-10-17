

t1_debug: t1.cu
	nvcc -gencode=arch=compute_35,code=\"sm_35,compute_35\" -rdc=true -lcudadevrt -O0 -G -g -lineinfo \
	-Xcompiler -rdynamic -Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -# -Xcompiler -ffloat-store \
	 -ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 -o $@ $<

t1_release: t1.cu
	nvcc -gencode=arch=compute_35,code=\"sm_35,compute_35\" -rdc=true -lcudadevrt -O3 \
	-Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -# -Xcompiler -ffloat-store \
	-ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 -o $@ $<

t1_ptx_with_code: t1.cu
	nvcc -gencode=arch=compute_35,code=\"sm_35,compute_35\" -rdc=true -lcudadevrt -O3 \
	-Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -# -Xcompiler -ffloat-store \
	-ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 \
	-lineinfo --source-in-ptx -ptx -o $@.ptx $<

t1_cubin: t1.cu 
	nvcc -arch=sm_35 -rdc=true -lcudadevrt -O3 \
	-Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -# -Xcompiler -ffloat-store \
	-Xptxas -preserve-relocs --nvlink-options -preserve-relocs -lineinfo \
	-ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 -cubin $<

t1_sass_with_code: t1.cubin
	nvdisasm -g -sf t1.cubin > t1_sass_w_code.sass

t1_sass_lrm: t1.cubin
	nvdisasm -lrm narrow -sf -g t1.cubin > t1_sass_lrm.sass

t1_sass_cfg: t1_cubin.cubin
	nvdisasm -cfg t1.cubin | dot -o t1_cfg.png -Tpng

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------

# include ../../cpu_common.mk

# t2: t2.cpp
# 	$(CC) $(DEFINES) $(CCFLAGS) $(INC) -o $@ $< $(LIBS)

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------

# t3: t3.cu
# 	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

t4: t4_debug t4_release

t4_debug: t4.cu 
	nvcc -arch=sm_35 -rdc=true -lcudadevrt -G -g \
	-Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -#  \
	-ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 -o $@ $<

# nvcc -gencode=arch=compute_35,code=\"sm_35,compute_35\"  \
# -rdc=true -lcudadevrt \
# -G -g -Xcompiler -rdynamic -Xptxas -g -Xcompiler -g3 -Xcompiler -Og \
# -Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -#  \
# -Xcompiler -ffloat-store -ccbin=/usr/bin/g++-4.9 -std=c++11 -m64  \
# -o $@ $<

t4_release: t4.cu
	nvcc -arch=sm_35 -rdc=true -lcudadevrt -O3 \
	-Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -#  \
	-ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 -o $@ $<

# nvcc -gencode=arch=compute_35,code=\"sm_35,compute_35\"\
# -rdc=true -lcudadevrt -O3 -Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -#  \
# -Xcompiler -ffloat-store -ccbin=/usr/bin/g++-4.9 -std=c++11 -m64  \
# -o $@ $<

t4_ptx: t4_debug_ptx t4_release_ptx

t4_debug_ptx: t4.cu
	nvcc -arch=sm_35 -rdc=true -lcudadevrt -G -g \
	-Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -#  \
	-ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 \
	-lineinfo --source-in-ptx -ptx -o $@.ptx $<

t4_release_ptx: t4.cu
	nvcc -arch=sm_35 -rdc=true -lcudadevrt -O3 \
	-Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -#  \
	-ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 \
	-lineinfo --source-in-ptx -ptx -o $@.ptx $<


#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------

t5_release: t5.cu
	nvcc -gencode=arch=compute_35,code=\"sm_35,compute_35\" -rdc=true -lcudadevrt -O3 \
	-Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -# -Xcompiler -ffloat-store \
	-ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 -o $@ $<

t5_ptx_with_code: t5.cu
	nvcc -gencode=arch=compute_35,code=\"sm_35,compute_35\" -rdc=true -lcudadevrt -O3 \
	-Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -# -Xcompiler -ffloat-store \
	-ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 \
	-lineinfo --source-in-ptx -ptx -o $@.ptx $<

t5_cubin: t5.cu 
	nvcc -arch=sm_35 -rdc=true -lcudadevrt -O3 \
	-Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -# -Xcompiler -ffloat-store \
	-Xptxas -preserve-relocs --nvlink-options -preserve-relocs -lineinfo \
	-ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 -cubin $<

t5_sass_with_code: t5.cubin
	nvdisasm -g -sf t5.cubin > t5_sass_w_code.sass

t5_sass: t5.cubin
	nvdisasm t5.cubin > t5_sass.sass

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------

t6_release: t6.cu
	nvcc -gencode=arch=compute_35,code=\"sm_35,compute_35\" -rdc=true -lcudadevrt -O3 \
	-Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -# -Xcompiler -ffloat-store \
	-ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 -o $@ $<

t6_profile: t6.cu
	nvcc -gencode=arch=compute_35,code=\"sm_35,compute_35\" -rdc=true -lcudadevrt -O3 \
	-lineinfo -Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -# -Xcompiler -ffloat-store \
	-ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 -o $@ $<

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------

t7_release: t7.cu
	nvcc -gencode=arch=compute_35,code=\"sm_35,compute_35\" -rdc=true -lcudadevrt -O3 \
	-Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -# -Xcompiler -ffloat-store \
	-ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 -o $@ $<

t7_profile: t7.cu
	nvcc -gencode=arch=compute_35,code=\"sm_35,compute_35\" -rdc=true -lcudadevrt -O3 \
	-lineinfo -Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -# -Xcompiler -ffloat-store \
	-ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 -o $@ $<

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------

t8_release_cdp: t8.cu
	nvcc -gencode=arch=compute_35,code=sm_35 -rdc=true --resource-usage -lcudadevrt -O3 \
	-Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -# -Xcompiler -ffloat-store \
	-ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 -o $@ $<

t8_profile_cdp: t8.cu
	nvcc -gencode=arch=compute_35,code=sm_35 -rdc=true --resource-usage -lcudadevrt -O3 \
	-lineinfo -Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -# -Xcompiler -ffloat-store \
	-ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 -o $@ $<

t8_release_nocdp: t8.cu
	nvcc -gencode=arch=compute_35,code=sm_35 --resource-usage -O3 \
	-Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -# -Xcompiler -ffloat-store \
	-ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 -o $@ $<

t8_profile_nocdp: t8.cu
	nvcc -gencode=arch=compute_35,code=sm_35 --resource-usage -O3 \
	-lineinfo -Xcompiler -Wall -Xcompiler -Wextra -Xcudafe -# -Xcompiler -ffloat-store \
	-ccbin=/usr/bin/g++-4.9 -std=c++11 -m64 -o $@ $<

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------

include ../../cpu_common.mk

t9: t9.cpp $(RD_DEPS) 
	$(CC) $(DEFINES) $(CCFLAGS) $(INC) -o $@ $< $(LIBS)

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------

clean:
	rm -rf t?_* t2 t3 t4 t5 t6 t7 t8 t9