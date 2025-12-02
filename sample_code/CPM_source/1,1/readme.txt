This are the sources from CPMUG disk #5 which have been recovered
so far. The sources on this disk have been deleted for unkown reasons
and it is difficult to get an original disk nowadays. If anyone has
an intact disk, with the CP/M sources, I would like a copy of the
missing files.

Notes from SIGM disk #15 about CPMUG disk #5:

CP/M SOURCE FILES

THE JUNE 1975 RELEASE OF CP/M IS IN PUBLIC DOMAIN.  THE PLM
AND ASSEMBLY FILES HERE ARE PART OF THAT RELEASE.

THE FULL RELEASE WAS:

                CCP.PLM
                BDOS.PLM
                PIP.PLM  (CALLED Z-PIP.PLM AS ADDED LATE)
                LOAD.PLM
                DUMP.ASM
                IOLIB.PLM

THESE ARE CERTIFIED BY GARY KILDALL TO BE AVAILABLE FOR
PUBLIC DISTRIBUTION FOR ANY PURPOSE WITHOUT RESTRICTION
[no author given]

The files:

	bdos-original.plm
	ccp-original.plm
	load-original.plm

are copies of the original files from this disk. The other files are
modified by me, to have working and usable programs on my virtual
Z80 system.

To build a disk from this files, that can be booted on z80sim,
compile ccp.plm and bdos.plm to hex files. Assemble boot.z80
with z80asm to a hex file.

An alternative boot loader is boot-rel.z80, to be assembled with
z80asm. This one relocates the boot loader to FF00H and sets up
the vectors in page zero, so that programs from CP/M 1.3 and 1.4,
namely DDT can be used.

Then copy the hex files to a CP/M 2 disk and use sysgen.sub, to write
the OS bits on the system tracks of an empty disk with CP/M 2 sysgen.

Compile load.plm and also put it on this disk (cpmtools) and boot it
then. The file hello.plm just is a small example, that can be compiled
to a hex file, and then loaded. The paper tape reader is implemented,
it is not necessary to copy it to the disk.

A ready to boot disk is available too from this site. This disk
is build with the boot loader boot-rel.z80, so CP/M 1.3 and 1.4
programs can be used on this system.

October 2007, Udo Munk
