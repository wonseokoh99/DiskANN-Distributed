#!/usr/bin/env python3
"""
DiskANN Graph Reader - Simple & Fast

요구사항:
1. 그래프 파일만 읽음 (데이터 파일 불필요)
2. CSV 출력: node_id, dataset_offset, degree, neighbors (정수만, node_ 없음)
3. 실시간 진행률 출력
4. 최대한 빠르게
"""

import struct
import argparse
import csv
import os
import sys
import time

# tqdm 시도 (실시간 출력용)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

def main():
    print("DiskANN Graph Reader 시작", flush=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_file', required=True, help='그래프 파일 경로')
    parser.add_argument('--data_offset', type=int, default=0, help='데이터셋 오프셋')
    parser.add_argument('--output_prefix', required=True, help='출력 파일 접두사')
    parser.add_argument('--save_detailed_edges', action='store_true', help='상세 엣지 파일 생성')
    parser.add_argument('--show_node', type=int, help='특정 노드 정보 조회')
    
    args = parser.parse_args()
    
    # 파일 확인
    if not os.path.exists(args.graph_file):
        print("그래프 파일이 존재하지 않습니다")
        sys.exit(1)
    
    start_time = time.time()
    
    # 그래프 파일 읽기
    print("그래프 파일 읽는 중...", flush=True)
    nodes = []
    
    with open(args.graph_file, 'rb') as f:
        # 헤더
        file_size, max_degree, start_node, num_frozen = struct.unpack('<QIIQ', f.read(24))
        print(f"그래프: max_degree={max_degree}, frozen_points={num_frozen}")
        
        # 노드 데이터
        node_id = 0
        while True:
            data = f.read(4)
            if not data:
                break
            
            num_neighbors = struct.unpack('<I', data)[0]
            
            neighbors = []
            if num_neighbors > 0:
                neighbor_data = f.read(num_neighbors * 4)
                if len(neighbor_data) != num_neighbors * 4:
                    break
                neighbors = list(struct.unpack(f'<{num_neighbors}I', neighbor_data))
            
            nodes.append((node_id, num_neighbors, neighbors))
            node_id += 1
            
            # 진행률
            if node_id % 100000 == 0:
                elapsed = time.time() - start_time
                rate = node_id / elapsed
                print(f"진행: {node_id:,} 노드 ({rate:.0f} nodes/sec)", flush=True)
    
    print(f"그래프 읽기 완료: {len(nodes):,} 노드")
    
    # 특정 노드 조회
    if args.show_node is not None:
        if args.show_node < len(nodes):
            node_id, degree, neighbors = nodes[args.show_node]
            print(f"\n노드 {args.show_node} 정보:")
            print(f"  degree: {degree}")
            print(f"  dataset_offset: {node_id + args.data_offset}")
            print(f"  neighbors: {neighbors[:10]}")
        return
    
    # CSV 생성
    csv_file = f"{args.output_prefix}_nodes_detailed.csv"
    print(f"CSV 생성: {csv_file}", flush=True)
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id', 'dataset_offset', 'degree', 'neighbors'])
        
        for node_id, degree, neighbors in tqdm(nodes, desc="CSV 작성"):
            dataset_offset = node_id + args.data_offset
            neighbors_str = ','.join(str(n) for n in neighbors) if neighbors else ''
            writer.writerow([node_id, dataset_offset, degree, neighbors_str])
    
    # 엣지 파일 (선택사항)
    if args.save_detailed_edges:
        edges_file = f"{args.output_prefix}_edges_detailed.csv"
        print(f"엣지 파일 생성: {edges_file}", flush=True)
        
        degrees = {node_id: degree for node_id, degree, _ in nodes}
        
        with open(edges_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source_node', 'target_node', 'source_degree', 'target_degree'])
            
            for node_id, degree, neighbors in tqdm(nodes, desc="엣지 생성"):
                for target_id in neighbors:
                    target_degree = degrees.get(target_id, 0)
                    writer.writerow([node_id, target_id, degree, target_degree])
    
    # 완료
    total_time = time.time() - start_time
    rate = len(nodes) / total_time
    
    print(f"\n완료!")
    print(f"총 시간: {total_time:.2f}초")
    print(f"처리 속도: {rate:.0f} nodes/sec")
    print(f"생성된 파일: {csv_file}")

if __name__ == "__main__":
    main()