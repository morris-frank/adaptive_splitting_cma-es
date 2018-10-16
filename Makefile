BDIR=$(CURDIR)

BCLASS=player28

help:
	@echo 'Makefile for the evolcomp assignment                                      '
	@echo '                                                                          '
	@echo 'Usage:                                                                    '
	@echo '   make build                  build the code                             '
	@echo '   make test                   run the eval                               '
	@echo '                                                                          '

build:
	javac -cp contest.jar $(BCLASS).java Jama/*.java Jama/util/*java altMatrix.java Vector.java player28_alt.java
	jar cmf MainClass.txt submission.jar $(BCLASS)*.class Jama/*class Jama/util/*class altMatrix.class Vector.class player28_alt.class

test:
	# java -Dverbose=false -jar testrun.jar -submission=$(BCLASS) -evaluation=BentCigarFunction -seed=$$RANDOM
	java -Dverbose=false -jar testrun.jar -submission=$(BCLASS) -evaluation=SchaffersEvaluation -seed=$$RANDOM
	# java -Dverbose=false -jar testrun.jar -submission=$(BCLASS) -evaluation=KatsuuraEvaluation -seed=$$RANDOM

.PHONY: help test build
