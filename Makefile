CXX      ?= g++
CXXFLAGS ?= -std=c++17
CPPFLAGS ?= -O3 -Wall -pedantic -I/home/miguelalcaniz/Escritorio/PACS/Proyecto/GitRepository/src
LDLIBS   ?= 
LINK.o := $(LINK.cc) # Use C++ linker.

DEPEND = make.dep

EXEC = main
SRCDIR = src
SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean distclean $(EXEC)

all: $(DEPEND) $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(OBJS) $(LDLIBS) -o $@

$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	$(RM) $(DEPEND)
	$(RM) $(SRCDIR)/*.o

distclean: clean
	$(RM) $(EXEC)
	$(RM) *.csv *.out *.bak *~

$(DEPEND): $(SRCS)
	@$(RM) $(DEPEND)
	@for file in $(SRCS); do \
	  $(CXX) $(CPPFLAGS) $(CXXFLAGS) -MM $$file >> $(DEPEND); \
	done

-include $(DEPEND)

# Debugging section
print-%: ; @echo $* = $($*)