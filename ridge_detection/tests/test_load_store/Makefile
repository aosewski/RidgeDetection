#***************************************************************************
# Author: Adam Rogowiec
#
#***************************************************************************/

include ../../gpu_common.mk

TEST_T1 = t1
TEST_LDG = test_ldg
TEST_TILE_LOAD_STORE_V4 = test_tile_load_store_v4
TEST_MISALIGNED = test_misaligned
TEST_STORE_FROM_SMEM = test_store_from_smem

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS =				$(RD_DEPS)

tile_load_store: $(TEST_TILE_LOAD_STORE_V4)  

#----------------------------------------------------------------------

$(TEST_T1)_cubin: $(TEST_T1).cu 
	$(NVCC) $(DEFINES) -gencode=arch=compute_35,code=sm_35 $(NVCCFLAGS) $(CPU_ARCH) \
	$(INC) -cubin $< 

$(TEST_T1)_sass_wc: $(TEST_T1)_cubin
	nvdisasm -g -sf $(TEST_T1).cubin > $(TEST_T1)_sass_w_code.sass

$(TEST_T1)_ptx_wc: $(TEST_T1).cu
	$(NVCC) $(DEFINES) -gencode=arch=compute_35,code=sm_35 $(NVCCFLAGS) $(CPU_ARCH) \
	$(INC) -lineinfo --source-in-ptx -ptx -o $(TEST_T1)_wc.ptx $<


#----------------------------------------------------------------------

$(TEST_LDG): $(TEST_LDG).cu 
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

#----------------------------------------------------------------------

$(TEST_TILE_LOAD_STORE_V4): $(TEST_TILE_LOAD_STORE_V4).cu 
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

#----------------------------------------------------------------------

$(TEST_MISALIGNED): $(TEST_MISALIGNED).cu 
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

#----------------------------------------------------------------------

$(TEST_STORE_FROM_SMEM): $(TEST_STORE_FROM_SMEM).cu 
	$(NVCC) $(DEFINES) $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INC) -o $@ $< $(LIBS)

#----------------------------------------------------------------------

clean:
	rm -f *.o $(TEST_TILE_LOAD_STORE_V4) $(TEST_LDG) $(TEST_MISALIGNED) $(TEST_STORE_FROM_SMEM) *.png $(TEST_T1)

clean-img:
	rm *.png

