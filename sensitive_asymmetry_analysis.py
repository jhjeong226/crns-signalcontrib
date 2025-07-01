# sensitive_asymmetry_analysis.py
"""
토양수분 비대칭성을 더 민감하게 감지하는 분석
"""

import numpy as np
import matplotlib.pyplot as plt

def enhanced_soil_moisture_interpolation(X, Y, measurements, sensitivity='high'):
    """
    더 민감한 토양수분 보간
    
    sensitivity options:
    - 'low': 기존 방법 (80m 영향반경, 30m 특성길이)
    - 'medium': 중간 (50m 영향반경, 20m 특성길이) 
    - 'high': 높음 (30m 영향반경, 10m 특성길이)
    - 'extreme': 극한 (20m 영향반경, 5m 특성길이)
    """
    
    # 민감도별 매개변수 설정
    params = {
        'low': {'influence_radius': 80, 'char_length': 30},
        'medium': {'influence_radius': 50, 'char_length': 20},
        'high': {'influence_radius': 30, 'char_length': 10},
        'extreme': {'influence_radius': 20, 'char_length': 5}
    }
    
    influence_radius = params[sensitivity]['influence_radius']
    char_length = params[sensitivity]['char_length']
    
    print(f"   🔧 민감도: {sensitivity} (영향반경: {influence_radius}m, 특성길이: {char_length}m)")
    
    moisture_field = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_grid, y_grid = X[i, j], Y[i, j]
            
            total_weight = 0
            weighted_sm = 0
            
            for k in range(len(measurements)):
                x_meas, y_meas, sm_meas = measurements[k]
                distance = np.sqrt((x_grid - x_meas)**2 + (y_grid - y_meas)**2)
                
                if distance < influence_radius:
                    # 더 급격한 가중치 감소
                    weight = np.exp(-distance / char_length)  # 지수적 감소
                    total_weight += weight
                    weighted_sm += weight * sm_meas
            
            if total_weight > 0:
                moisture_field[i, j] = weighted_sm / total_weight
            else:
                # 기본값: 전체 평균
                moisture_field[i, j] = np.mean(measurements[:, 2])
    
    return moisture_field

def calculate_directional_asymmetry(moisture_field, X, Y):
    """방향별 토양수분 비대칭성 계산"""
    
    center_i, center_j = moisture_field.shape[0] // 2, moisture_field.shape[1] // 2
    
    # 방향별 영역 정의 (반원)
    directions = {}
    
    for i in range(moisture_field.shape[0]):
        for j in range(moisture_field.shape[1]):
            x, y = X[i, j], Y[i, j]
            distance = np.sqrt(x**2 + y**2)
            
            if distance <= 100:  # 100m 내부만
                if y > 0:  # 북쪽
                    if 'North' not in directions:
                        directions['North'] = []
                    directions['North'].append(moisture_field[i, j])
                elif y < 0:  # 남쪽
                    if 'South' not in directions:
                        directions['South'] = []
                    directions['South'].append(moisture_field[i, j])
                    
                if x > 0:  # 동쪽
                    if 'East' not in directions:
                        directions['East'] = []
                    directions['East'].append(moisture_field[i, j])
                elif x < 0:  # 서쪽
                    if 'West' not in directions:
                        directions['West'] = []
                    directions['West'].append(moisture_field[i, j])
    
    # 방향별 평균 계산
    directional_avg = {}
    for direction, values in directions.items():
        directional_avg[direction] = np.mean(values)
    
    return directional_avg

def run_sensitivity_analysis():
    """민감도별 분석 실행"""
    
    print("🔬 토양수분 비대칭성 민감도 분석")
    print("=" * 50)
    
    # 실제 센서 데이터 (콘솔 결과에서)
    sensor_data = {
        'E50': (50, 0, 0.247), 'N100': (0, 100, 0.229), 'N50': (0, 50, 0.202),
        'NE50': (35, 35, 0.236), 'NE75': (53, 53, 0.227), 'NW100': (-71, 71, 0.270),
        'NW25': (-18, 18, 0.280), 'S25': (0, -25, 0.323), 'S75': (0, -75, 0.248),
        'SE100': (71, -71, 0.279), 'SE75': (53, -53, 0.222), 'SW35': (-25, -25, 0.213)
    }
    
    # CRNS 중심 추가
    measurements = [[0, 0, 0.25]]  
    for sensor, (x, y, sm) in sensor_data.items():
        measurements.append([x, y, sm])
    measurements = np.array(measurements)
    
    # 격자 생성 (작은 영역으로 고해상도)
    grid_size = 201
    grid_range = 120
    x_range = np.linspace(-grid_range, grid_range, grid_size)
    y_range = np.linspace(-grid_range, grid_range, grid_size)
    X, Y = np.meshgrid(x_range, y_range)
    
    # 민감도별 분석
    sensitivities = ['low', 'medium', 'high', 'extreme']
    results = {}
    
    for sensitivity in sensitivities:
        print(f"\n🔍 {sensitivity.upper()} 민감도 분석:")
        
        # 토양수분 보간
        moisture_field = enhanced_soil_moisture_interpolation(X, Y, measurements, sensitivity)
        
        # 방향별 평균 계산
        directional_avg = calculate_directional_asymmetry(moisture_field, X, Y)
        
        # 비대칭성 지표 계산
        north_south_diff = directional_avg['North'] - directional_avg['South']
        east_west_diff = directional_avg['East'] - directional_avg['West']
        max_diff = max(directional_avg.values()) - min(directional_avg.values())
        
        results[sensitivity] = {
            'directional_avg': directional_avg,
            'north_south_diff': north_south_diff,
            'east_west_diff': east_west_diff,
            'max_diff': max_diff,
            'moisture_field': moisture_field
        }
        
        print(f"   방향별 평균 토양수분:")
        for direction, avg in directional_avg.items():
            print(f"     {direction:5}: {avg:.3f}")
        print(f"   남북 차이: {north_south_diff:+.3f}")
        print(f"   동서 차이: {east_west_diff:+.3f}")
        print(f"   최대 차이: {max_diff:.3f}")
    
    # 시각화
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for idx, sensitivity in enumerate(sensitivities):
        # 상단: 토양수분 분포
        ax1 = axes[0, idx]
        im = ax1.imshow(results[sensitivity]['moisture_field'], 
                       extent=[-grid_range, grid_range, -grid_range, grid_range],
                       origin='lower', cmap='Blues', vmin=0.20, vmax=0.33)
        
        # 센서 위치 표시
        for sensor, (x, y, sm) in sensor_data.items():
            ax1.scatter(x, y, c=sm, s=100, cmap='Blues', vmin=0.20, vmax=0.33,
                       edgecolors='black', linewidth=2)
        ax1.scatter(0, 0, c='red', s=200, marker='s')
        
        ax1.set_title(f'{sensitivity.upper()} Sensitivity\nSoil Moisture Field')
        ax1.set_xlabel('Distance East (m)')
        if idx == 0:
            ax1.set_ylabel('Distance North (m)')
        
        # 하단: 비대칭성 지표
        ax2 = axes[1, idx]
        directions = list(results[sensitivity]['directional_avg'].keys())
        values = list(results[sensitivity]['directional_avg'].values())
        
        bars = ax2.bar(directions, values, color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'])
        ax2.set_title(f'Directional Asymmetry\nMax diff: {results[sensitivity]["max_diff"]:.3f}')
        ax2.set_ylabel('Avg Soil Moisture')
        ax2.set_ylim(0.22, 0.28)
        
        # 값 라벨
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('sensitivity_asymmetry_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 결과 요약
    print(f"\n📊 민감도별 비대칭성 요약:")
    for sensitivity in sensitivities:
        max_diff = results[sensitivity]['max_diff']
        ns_diff = abs(results[sensitivity]['north_south_diff'])
        ew_diff = abs(results[sensitivity]['east_west_diff'])
        print(f"   {sensitivity:8}: 최대차이={max_diff:.3f}, 남북차이={ns_diff:.3f}, 동서차이={ew_diff:.3f}")
    
    return results

if __name__ == "__main__":
    results = run_sensitivity_analysis()
    print("\n✅ 민감도 분석 완료!")
    print("📁 결과 저장: sensitivity_asymmetry_analysis.png")