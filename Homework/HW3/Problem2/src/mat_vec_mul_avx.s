	.file	"mat_vec_mul_avx.c"
	.text
	.p2align 4
	.globl	CLOCK
	.type	CLOCK, @function
CLOCK:
.LFB5681:
	.cfi_startproc
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
	movl	$1, %edi
	movq	%rsp, %rsi
	call	clock_gettime
	imulq	$1000, (%rsp), %rax
	vxorps	%xmm1, %xmm1, %xmm1
	vcvtsi2sdq	8(%rsp), %xmm1, %xmm0
	vcvtsi2sdq	%rax, %xmm1, %xmm1
	vfmadd132sd	.LC0(%rip), %xmm1, %xmm0
	addq	$24, %rsp
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE5681:
	.size	CLOCK, .-CLOCK
	.p2align 4
	.globl	matrix_vector_multiplication
	.type	matrix_vector_multiplication, @function
matrix_vector_multiplication:
.LFB5682:
	.cfi_startproc
	movq	%rdi, %rcx
	vxorps	%xmm1, %xmm1, %xmm1
	leaq	8192(%rdx), %rdi
.L6:
	movl	$0x00000000, (%rdx)
	xorl	%eax, %eax
	vmovaps	%xmm1, %xmm0
	.p2align 4,,10
	.p2align 3
.L5:
	vmovss	(%rcx,%rax), %xmm2
	vfmadd231ss	(%rsi,%rax), %xmm2, %xmm0
	addq	$4, %rax
	vmovss	%xmm0, (%rdx)
	cmpq	$8192, %rax
	jne	.L5
	addq	$4, %rdx
	addq	$8192, %rcx
	cmpq	%rdi, %rdx
	jne	.L6
	ret
	.cfi_endproc
.LFE5682:
	.size	matrix_vector_multiplication, .-matrix_vector_multiplication
	.p2align 4
	.globl	matrix_vector_multiplication_avx512f
	.type	matrix_vector_multiplication_avx512f, @function
matrix_vector_multiplication_avx512f:
.LFB5683:
	.cfi_startproc
	movq	%rdx, %rcx
	vxorps	%xmm4, %xmm4, %xmm4
	movq	%rdi, %rdx
	leaq	16777216(%rdi), %rdi
.L10:
	xorl	%eax, %eax
	vxorps	%xmm0, %xmm0, %xmm0
	.p2align 4,,10
	.p2align 3
.L11:
	vmovups	(%rdx,%rax), %zmm6
	vfmadd231ps	(%rsi,%rax), %zmm6, %zmm0
	addq	$64, %rax
	cmpq	$8192, %rax
	jne	.L11
	vaddss	%xmm4, %xmm0, %xmm3
	vshufps	$85, %xmm0, %xmm0, %xmm5
	addq	$4, %rcx
	vshufps	$255, %xmm0, %xmm0, %xmm2
	vextractf128	$0x1, %ymm0, %xmm1
	addq	$8192, %rdx
	vaddss	%xmm3, %xmm5, %xmm5
	vunpckhps	%xmm0, %xmm0, %xmm3
	vextracti64x4	$0x1, %zmm0, %ymm0
	vaddss	%xmm5, %xmm3, %xmm3
	vaddss	%xmm3, %xmm2, %xmm2
	vshufps	$85, %xmm1, %xmm1, %xmm3
	vaddss	%xmm2, %xmm1, %xmm2
	vaddss	%xmm2, %xmm3, %xmm3
	vunpckhps	%xmm1, %xmm1, %xmm2
	vshufps	$255, %xmm1, %xmm1, %xmm1
	vaddss	%xmm3, %xmm2, %xmm2
	vaddss	%xmm2, %xmm1, %xmm1
	vaddss	%xmm1, %xmm0, %xmm3
	vshufps	$85, %xmm0, %xmm0, %xmm1
	vaddss	%xmm3, %xmm1, %xmm1
	vunpckhps	%xmm0, %xmm0, %xmm3
	vaddss	%xmm1, %xmm3, %xmm3
	vshufps	$255, %xmm0, %xmm0, %xmm1
	vextractf128	$0x1, %ymm0, %xmm0
	vshufps	$85, %xmm0, %xmm0, %xmm2
	vaddss	%xmm3, %xmm1, %xmm1
	vaddss	%xmm1, %xmm0, %xmm1
	vaddss	%xmm1, %xmm2, %xmm2
	vunpckhps	%xmm0, %xmm0, %xmm1
	vshufps	$255, %xmm0, %xmm0, %xmm0
	vaddss	%xmm2, %xmm1, %xmm1
	vaddss	%xmm1, %xmm0, %xmm0
	vmovss	%xmm0, -4(%rcx)
	cmpq	%rdx, %rdi
	jne	.L10
	vzeroupper
	ret
	.cfi_endproc
.LFE5683:
	.size	matrix_vector_multiplication_avx512f, .-matrix_vector_multiplication_avx512f
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC3:
	.string	"Matrix-Vector Multiplication Duration Without AVX: %f ms\n"
	.align 8
.LC4:
	.string	"Matrix-Vector Multiplication Duration With AVX: %f ms\n"
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC5:
	.string	"AVX Speedup: %f\n"
.LC6:
	.string	"Results Do Not Match: \t"
	.section	.rodata.str1.8
	.align 8
.LC7:
	.string	"result[%d] = %f, result_avx[%d] = %f\n"
	.section	.rodata.str1.1
.LC8:
	.string	"Results Match"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB5684:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-64, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	movq	%rsp, %rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x70,0x6
	.cfi_escape 0x10,0xc,0x2,0x76,0x78
	leaq	-16777264(%rbp), %rcx
	leaq	-16769072(%rbp), %rdx
	pushq	%rbx
	leaq	8144(%rbp), %rsi
	subq	$16801880, %rsp
	.cfi_escape 0x10,0x3,0x2,0x76,0x68
	vbroadcastss	.LC9(%rip), %zmm0
.L16:
	movq	%rcx, %rax
	.p2align 4,,10
	.p2align 3
.L17:
	vmovaps	%zmm0, (%rax)
	addq	$64, %rax
	cmpq	%rdx, %rax
	jne	.L17
	leaq	8192(%rax), %rdx
	addq	$8192, %rcx
	cmpq	%rsi, %rdx
	jne	.L16
	leaq	-16801840(%rbp), %rax
	leaq	-16793648(%rbp), %rbx
.L19:
	vmovaps	%zmm0, (%rax)
	addq	$64, %rax
	cmpq	%rax, %rbx
	jne	.L19
	xorl	%eax, %eax
	vzeroupper
	call	CLOCK
	movq	%rbx, %rdx
	leaq	-16801840(%rbp), %rsi
	leaq	-16777264(%rbp), %rdi
	vmovsd	%xmm0, -16801848(%rbp)
	call	matrix_vector_multiplication
	xorl	%eax, %eax
	call	CLOCK
	movl	$.LC3, %edi
	movl	$1, %eax
	vsubsd	-16801848(%rbp), %xmm0, %xmm1
	vmovsd	%xmm1, -16801856(%rbp)
	vmovsd	%xmm1, %xmm1, %xmm0
	call	printf
	xorl	%eax, %eax
	call	CLOCK
	leaq	-16785456(%rbp), %rdx
	leaq	-16801840(%rbp), %rsi
	leaq	-16777264(%rbp), %rdi
	vmovsd	%xmm0, -16801848(%rbp)
	call	matrix_vector_multiplication_avx512f
	xorl	%eax, %eax
	call	CLOCK
	movl	$.LC4, %edi
	movl	$1, %eax
	vsubsd	-16801848(%rbp), %xmm0, %xmm0
	vmovsd	%xmm0, -16801848(%rbp)
	call	printf
	movl	$.LC5, %edi
	movl	$1, %eax
	vmovsd	-16801848(%rbp), %xmm0
	vmovsd	-16801856(%rbp), %xmm1
	vdivsd	%xmm0, %xmm1, %xmm0
	call	printf
	xorl	%eax, %eax
	jmp	.L23
.L20:
	addq	$1, %rax
	cmpq	$2048, %rax
	je	.L29
.L23:
	vmovss	(%rbx,%rax,4), %xmm0
	movl	%eax, %r12d
	vmovss	-16785456(%rbp,%rax,4), %xmm1
	vucomiss	%xmm1, %xmm0
	jp	.L24
	je	.L20
.L24:
	movl	$.LC6, %edi
	xorl	%eax, %eax
	vmovss	%xmm1, -16801856(%rbp)
	vmovss	%xmm0, -16801848(%rbp)
	call	printf
	movl	%r12d, %edx
	movl	%r12d, %esi
	movl	$.LC7, %edi
	vmovss	-16801848(%rbp), %xmm0
	movl	$2, %eax
	vmovss	-16801856(%rbp), %xmm1
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vcvtss2sd	%xmm1, %xmm1, %xmm1
	call	printf
.L22:
	addq	$16801880, %rsp
	xorl	%eax, %eax
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
.L29:
	.cfi_restore_state
	movl	$.LC8, %edi
	call	puts
	jmp	.L22
	.cfi_endproc
.LFE5684:
	.size	main, .-main
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC0:
	.long	-1598689907
	.long	1051772663
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC9:
	.long	1065353216
	.ident	"GCC: (GNU) 11.4.1 20230605 (Red Hat 11.4.1-2)"
	.section	.note.GNU-stack,"",@progbits
