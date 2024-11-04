SUBDIRS := c

all: build
build:
	bindgen c/addmixed_op.hpp -o src/bindings/addmixed_op.r

test:
	cargo test -- --nocapture 2>&1 | tee c/test/data.log




.PHONY: clean
clean: 
	$(MAKE) -C $(SUBDIRS) clean
	cargo clean
	rm c/*.log
	rm c/test/*.log



