1. 레지스터 타일링 적용
2. Double Buffering
3. warp level parallelism 적용

2. 메모리 전송 최적화
2.1 Weight 사전 로드 (이미 일부 구현됨)
현재 898회 HTOD 전송 → 인퍼런스 전 1회로 줄여야 함
ModelLoader에서 모든 weight를 GPU에 캐시
2.2 Pinned Memory 사용
cudaHostAlloc(&h_data_, size_ * sizeof(float), cudaHostAllocDefault);
현재 12.16 GB/s → Pinned 사용 시 ~14 GB/s 이상 가능
2.3 비동기 전송 + 스트림 활용
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);

3. 커널 퓨전
3.1 이미 구현된 것
silu_mul_kernel - SiLU + Mul 퓨전 ✅
3.2 추가 퓨전 기회
// RMSNorm + Linear 퓨전
// Attention: Scale + Softmax 퓨전
// 출력: Add (residual) + RMSNorm 퓨전

4. Attention 최적화
커널	시간	호출 수
batched_attn_scores	2.54ms	24
batched_attn_output	0.60ms	24
batched_softmax	0.46ms	24
Flash Attention 적용
현재 구현은 Q×K^T 전체 계산 후 softmax
Flash Attention: 타일 기반으로 메모리 접근 최적화
메모리 O(N²) → O(N)