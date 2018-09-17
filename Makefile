BDIR=$(CURDIR)

LOCALURL=http://localhost/

help:
	@echo 'Makefile for the evolcomp assignment                                      '
	@echo '                                                                          '
	@echo 'Usage:                                                                    '
	@echo '   make build                  build the code                             '
	@echo '   make test                   run the eval                               '
	@echo '                                                                          '

build:
	javac -cp contest.jar player28.java
	jar cmf MainClass.txt submission.jar player28.class

test:
	java -jar testrun.jar -submission=player28 -evaluation=BentCigarFunction -seed=1

.PHONY: help test build
