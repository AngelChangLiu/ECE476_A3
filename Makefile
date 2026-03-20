.PHONY: all build clean rebuild submit archive info help $(DIRS) \
        build-% clean-%

DIRS ?= saxpy scan render
ARCHIVE ?= asst3.tar.gz

all: build

build: info $(addprefix build-,$(DIRS))

clean: $(addprefix clean-,$(DIRS))

rebuild: clean build

submit: clean archive

archive:
	tar -czvf $(ARCHIVE) $(DIRS)

info:
	@echo "GCC version: $$(gcc --version | head -n 1)"
	@echo "NVCC version: $$(nvcc --version | grep release | awk '{print $$6}' | cut -d',' -f1)"

help:
	@echo "Top-level targets:"
	@echo "  make                 Build all assignment parts"
	@echo "  make build           Build all assignment parts"
	@echo "  make clean           Clean all assignment parts"
	@echo "  make rebuild         Clean then build all assignment parts"
	@echo "  make submit          Clean and create $(ARCHIVE)"
	@echo "  make archive         Create $(ARCHIVE) without cleaning"
	@echo "  make saxpy           Build only saxpy"
	@echo "  make scan            Build only scan"
	@echo "  make render          Build only render"
	@echo "Variables:"
	@echo "  DIRS='saxpy scan render'   Subprojects to operate on"
	@echo "  ARCHIVE=asst3.tar.gz       Submission archive name"

$(DIRS): %: build-%

build-%:
	$(MAKE) -C $* -j

clean-%:
	$(MAKE) -C $* clean
