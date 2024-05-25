##### 코드 처음 실행 후 #####

File "PTF01.py", line 62:
def compute_tetration_divergence(c, nx, ny, max_iter, escape_radius):
    <source elided>
            z = c_val
            for k in prange(max_iter):
            ^

##### 위와 같이 메세지가 뜰텐데 무시하세요 #####

이 코드는 DMT Park의 Power Tower Fractal 코드에 기능을 추가하고 죄적화한 것입니다.
추가 기능은 아래와 같습니다.

좌클릭: 클릭한 곳 확대, 클릭 후 기다리면 확대됩니다.
우클릭: 전으로가기, 클릭을 하면 이전에 봤던 곳으로 가집니다.

확대 배율은 코드에서 수정할 수 있습니다.

최적화 현황

시간 측정 조건
Python 버전 : python 3.9.10
PC 사양
 - CPU	Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz   2.30 GHz
 - RAM	16.0GB(15.8GB 사용 가능)
 - GPU Intel(R) UHD Graphics

Time.time() 사용, 초를 hh:mm:ss로 바꾸기 위해 datetime.timedelta와 시간 포멧팅 함수 사용

Parameters
x0 = 0
y0 = 0 #(x0,y0)
eps = 5e0
eps_y = eps * (9/16)
n = 500
nx, ny = n, int(n*(9/16))
max_iter = 500
escape_radius = 1e+10

기존 코드:
00 hours : 01 minutes : 24.436294 seconds

최적화 후 코드:
00 hours : 00 minutes : 04.717862 seconds

포멧팅 함수

def seconds_to_hms(seconds):
    # Create a timedelta object from the given seconds
    td = timedelta(seconds=seconds)
    # Get the total number of seconds and the fractional part
    total_seconds = int(td.total_seconds())
    fractional_seconds = td.total_seconds() - total_seconds
    # Calculate hours, minutes, and seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60 + fractional_seconds
    # Format the time in hh:mm:ss.ssssss
    return f"{hours:02} hours : {minutes:02} minutes : {seconds:09.6f} seconds"