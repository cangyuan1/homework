#一、装好库
import sys
import subprocess

当前python解释器=sys.executable
需要的库=["numpy","pandas","matplotlib","statsmodels","scipy"]
for 库 in 需要的库:
    subprocess.check_call([当前python解释器,"-m","pip","install",库])

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
np.random.seed(1)
#一些工具函数
def ab(a,b):
    '''用来交替Y1和Y0,后面弄长面板数据时要用到，但是最后发现长面板数据(df)没有用到，都是用的宽的（df_forPSM）'''
    c=[]
    for i in range(0,len(a)):
       c.append(a[i])
       c.append(b[i])
    return c

def add_core(df_forPSM):
    '''df_forPSM是个dataFrame结构的表，这个函数用来给这个表算倾向值得分，然后弄成一列加在原表上'''
    x=df_forPSM[["Xi"]].copy()
    x=sm.add_constant(x)
    y=df_forPSM["Di"]
    model=sm.Logit(y,x).fit(disp=0)
    score=model.predict(x)
    df_forPSM["pScore"]=score
    return df_forPSM

def get_pairs(df_forPSM_withScore):
    '''使用最邻近匹配法（不放回），卡尺为0.03,对已经有倾向值得分的dataframe表，返回一个元组列表，里面装的是元组，每个元组是（处理组个体，匹配到的对照组个体）'''
    treatG=df_forPSM_withScore[df_forPSM_withScore["Di"]==1].copy()
    controllG=df_forPSM_withScore[df_forPSM_withScore["Di"]==0].copy()
    controll_pool=controllG.copy()
    matched=[]
    for i,row in treatG.iterrows():
        if controll_pool.empty:
            break
        diff=(controll_pool["pScore"]-row["pScore"]).abs()
        if diff.min()<0.03:
            friend=diff.idxmin()
            matched.append((row["id"],controll_pool.loc[friend,"id"]))
            controll_pool=controll_pool.drop(index=friend)
    return matched

def compute_ATT(df_forPSM,matched_pairs):
    '''用来根据配对的对子和df_forPSM来计算PSM法的ATT'''
    treat_id=[x[0] for x in matched_pairs]
    controll_id=[x[1] for x in matched_pairs]
    treat_Y1=df_forPSM[df_forPSM["id"].isin(treat_id)]["Yi1"]
    controll_Y1=df_forPSM[df_forPSM["id"].isin(controll_id)]["Yi1"]
    ATT=treat_Y1.mean()-controll_Y1.mean()
    return ATT

def compute_PSMDID(df_forPSM,matched_pairs):
    '''用来根据配对的对子和df_forPSM来计算PSM-DID法的ATT'''
    diff=[]
    for t_id,c_id in matched_pairs:
        t_row=df_forPSM[df_forPSM["id"]==t_id].iloc[0]
        c_row=df_forPSM[df_forPSM["id"]==c_id].iloc[0]
        diff.append((t_row["Yi1"]-t_row["Yi0"])-(c_row["Yi1"]-c_row["Yi0"]))
    return np.mean(diff)

#二、布置场景的函数

#   二、1.场景A——处理组和控制组满足无条件平行趋势
#       我个人理解：也就是没有随时间变化的混淆变量。影响潜在结果的数值的只有四类因素：个体固定效应，时间固定效应、处理效应、随机误差。
#       也就是说,对于每一个个体：Yit =  a_i +  l_t   +   tao*D_i*post_t   +   e_it
#       由于至少有一个协变量，则场景A模型：Yit=a_i0+β*X_i+l_t+tao_i*D_i*post_t+e_it
#       我的理解：协变量在这个场景中代表了一部分可观测的不随时间变化的混淆因素，剩余的由a_i0解释，即a_i0+X_i=a_i
#       协变量跟D_i的关系是：协变量影响D_i，即logit(p)=m+n*X_i+残差,D_i~B(1,p)
#       协变量跟a_i0的关系是：协变量影响a_i0,即a_i0=h+g*Xi+残差
def Get_A_paneldata(N):
    '''生成一次场景A的面板数据，只有一个协变量（后同），返回一个列表，有两个元素（后面的场景也一样）（第二个元素是后加的，但是又不舍得删前面那个元素，所以就直接设定成返回一个列表），第一个元素是长面板数据，也就是每个个体占两行（处理前和处理后），列有id post_t Xi Di Yit，第二个元素是扁的面板数据，列有id Xi Di Yi1 Yi0。N是样本量，T是期数，至少两期，tao是真实处理效应，l（是个列表，长度跟T的大小对应，也就是有几期就有几个时间固定效应）是时间固定效应，……算了，参数太多了，就不弄成形参了，写死在函数里面算了'''
    T=2
    tao=3#tao是真实处理效应
    l=[0,2]#l是lamda,即时间固定效应（或者说时间冲击）
    X=np.random.normal(0,1,N)#X是协变量，一个列表，所有个体的协变量Xi值都在这里
    #print("X",X)
    logitp=1+0.5*X
    #print("logitp",logitp)
    p=1/(1+np.exp(-logitp))
    D=np.random.binomial(1,p)#D是是否处理，一个列表，一一对应的，所有的都在这
    #print("D",D)
    #print(D.mean())
    a=1+0.5*X+np.random.normal(0,0.5)#这是总个体固定效应列表，与协变量列表有关
    e_pre=np.random.normal(0,1,N)#干预前每个人的残差，独立于其他所有东西
    e_post=np.random.normal(0,1,N)#干预后的每个人的残差，也是完全独立的
    Y0=a+l[0]+e_pre
    Y1=a+l[1]+tao*D+e_post
    #用pandas弄成表格
    df=pd.DataFrame({
        "id":np.repeat(np.arange(N),T),
        "post":np.tile([0,1],N),
        "Xi":np.repeat(X,T),
        "Di":np.repeat(D,T),
        "Yit":ab(Y0,Y1)
    })
    df_for_PSM=pd.DataFrame({
        "id":np.arange(N),
        "Xi":X,
        "Di":D,
        "Yi0":Y0,
        "Yi1":Y1
    })
    #print("df",df)
    return [df,df_for_PSM]
#   2.场景B——处理组和对照组趋势差异与处理前协变量Xi有关，匹配后条件平行趋势成立
#       我个人理解：也就是有随时间变化的混淆因素，但是全部都被协变量解释了，也就是有X_it，或者写成X_i*post_t（因为只有两期）,为了简化，不妨假设观测到的协变量恰好就是所有随时间变化的混淆因素，它与个体固定效应无关
#       也就是说，对于每个个体：Yit =  a_i +  l_t   +   tao*D_i*post_t   +  β*X_i*post_t  +  e_it
def Get_B_paneldata(N):
    '''生成一次场景B的面板数据，时变混淆完全被协变量Xi解释，控制了Xi即可平行'''
    T=2#期数
    tao=3
    l=[0,2]#真实时间固定效应
    Beta=1.5
    X=np.random.normal(0,1,N)
    logitp=1+0.5*X
    p=1/(1+np.exp(-logitp))
    D=np.random.binomial(1,p)
    #print(D.mean())
    a=np.random.normal(0,1,N)
    e_pre=np.random.normal(0,1,N)
    e_post=np.random.normal(0,1,N)
    Y0=a+l[0]+e_pre
    Y1=a+l[1]+tao*D+Beta*X+e_post
    df=pd.DataFrame({
        "id":np.repeat(np.arange(N),T),
        "post":np.tile([0,1],N),
        "Xi":np.repeat(X,T),
        "Di":np.repeat(D,T),
        "Yit":ab(Y0,Y1)
    })
    df_forPSM=pd.DataFrame({
        "id":np.arange(N),
        "Xi":X,
        "Di":D,
        "Yi0":Y0,
        "Yi1":Y1
    })
    #print(df)
    return [df,df_forPSM]
#   3.场景C——存在不可观测的时间变化混淆
#       我个人理解，存在时变混淆Xi*post_t，但是不能完全观测到，能观测到的只能解释部分时变混淆Mi，也就是Xi与Mi相关，但是真实模型里用的是Xi
#       也就是说，对于每个个体，真实模型为：Yit =  a_i +  l_t   +   tao*D_i*post_t   +  β*X_i*post_t  +  e_it
#       其中，Mi=a+bXi+残差。
def Get_C_paneldata(N):
    '''生成一次场景C的面板数据，协变量Xi只能解释一部分的时变混淆，仍有其他未观测到的时变混淆'''
    T=2
    l=[0,2]
    tao=3
    Beta=1.5
    X=np.random.normal(0,1,N)
    logitp=1+0.5*X
    p=1/(1+np.exp(-logitp))
    D=np.random.binomial(1,p)
    M=1+0.5*X+np.random.normal(0,1,N)
    e_pre=np.random.normal(0,1,N)
    e_post=np.random.normal(0,1,N)
    a=np.random.normal(0,1,N)
    Y0=a+l[0]+e_pre
    Y1=a+l[1]+D*tao+Beta*X+e_post
    df=pd.DataFrame({
        "id":np.repeat(np.arange(N),T),
        "post":np.tile([0,1],N),
        "Xi":np.repeat(M,T),
        "Di":np.repeat(D,T),
        "Yit":ab(Y0,Y1)
    })
    #print(df)
    df_forPSM=pd.DataFrame({
        "id":np.arange(N),
        "Xi":M,
        "Di":D,
        "Yi0":Y0,
        "Yi1":Y1
    })
    return [df,df_forPSM]
#Get_C_paneldata(5)
#   4.场景D——共同支持不足
#   大背景还是满足无条件平行趋势假设，唯独增强了Xi对D的解释，并且让Xi在对照组和实验组的分布离散，对照组的Xi均值为0.3，处理组为0.7
def Get_D_paneldata(N):
    '''得到场景D的面板数据——满足无条件平行趋势假设，但是共同支持不足，有一个可观测协变量'''
    T=2
    tao=3
    l=[0, 2]
    D=np.random.binomial(1, 0.5, N)  # 临时暂时先随机分配处理状态
    #不是很偏的版本
    '''
    X=np.where(D==1,
                 np.random.normal(0.7, 0.8, N),
                 np.random.normal(0.3, 0.8, N))
    '''
    #偏一点的版本
    X = np.where(D == 1,
                 np.random.normal(1.5, 0.5, N),
                 np.random.normal(-1.5, 0.5, N))
    logit_p=-0.5+3*X
    p_true=1/(1+np.exp(-logit_p))
    D=np.random.binomial(1,p_true)
    a=1+0.5*X+np.random.normal(0,0.5,N)
    e_pre=np.random.normal(0,1,N)
    e_post=np.random.normal(0,1,N)

    Y0=a+l[0]+e_pre
    Y1=a+l[1]+tao*D+e_post
    df=pd.DataFrame({
        "id":np.repeat(np.arange(N), T),
        "post":np.tile([0, 1], N),
        "Xi":np.repeat(X, T),
        "Di":np.repeat(D, T),
        "Yit":ab(Y0, Y1)
    })
    df_forPSM=pd.DataFrame({
        "id":np.arange(N),
        "Xi":X,
        "Di":D,
        "Yi0":Y0,
        "Yi1":Y1
    })
    return [df, df_forPSM]

def analyze(模拟次数,场景,N):
    '''进行某个场景进行500次（模拟次数=500时）PSM分析和PSM-DID分析和DID分析和控制协变量的DID分析，返回一个包含四个元素的列表（每个元素对应一个方法，是一个列表，装着该方法下500次模拟得到的估计值）'''
    att_list=[]
    PSMDID_list=[]
    DID_list=[]
    DID_cov_list=[]
    for i in range(0,模拟次数):
        if 场景 == "A":
            l=Get_A_paneldata(N)
        elif 场景=="B":
            l=Get_B_paneldata(N)
        elif 场景=="C":
            l=Get_C_paneldata(N)
        elif 场景=="D":
            l=Get_D_paneldata(N)
        #PSM
        add_core(l[1])
        p=get_pairs(l[1])
        att=compute_ATT(l[1],p)
        att_list.append(att)
        #PSM-DID
        psmdid_att=compute_PSMDID(l[1],p)
        PSMDID_list.append(psmdid_att)
        #DID
        l[1]["delta"]=l[1]["Yi1"]-l[1]["Yi0"]
        DID_att=np.mean(l[1][l[1]["Di"]==1]["delta"])-np.mean(l[1][l[1]["Di"]==0]["delta"])
        DID_list.append(DID_att)
        #控制协变量的DID
        model = smf.ols('delta ~ Di + Xi', data=l[1]).fit()
        did_cov = model.params['Di']
        DID_cov_list.append(did_cov)
    return [att_list,PSMDID_list,DID_list,DID_cov_list]

def more_analyze(analyze_result):
    '''计算给定的analyze()函数的结果的的统计量（均值、偏差、RMSE、标准差），也就是各种估计方法的估计值的这些统计量'''
    统计量={}
    psm_arr = np.array(analyze_result[0])
    psmdid_arr = np.array(analyze_result[1])
    did_arr = np.array(analyze_result[2])
    did_cov_arr = np.array(analyze_result[3])
    #各种方法的估计值均值
    统计量["纯PSM估计值均值"]=psm_arr.mean()
    统计量["PSM-DID估计值均值"]=psmdid_arr.mean()
    统计量["纯DID估计值均值"]=did_arr.mean()
    #各种方法的估计值Bias
    统计量["纯PSM的Bias"]=统计量["纯PSM估计值均值"]-3
    统计量["PSM-DID的Bias"] = 统计量["PSM-DID估计值均值"] - 3
    统计量["纯DID的Bias"] = 统计量["纯DID估计值均值"] - 3
    #各种方法的估计值RMSE
    统计量["纯PSM的RMSE"] = np.sqrt(np.mean((psm_arr-3)**2))
    统计量["PSM-DID的RMSE"] = np.sqrt(np.mean((psmdid_arr-3)**2))
    统计量["纯DID的RMSE"] = np.sqrt(np.mean((did_arr-3)**2))
    #各种方法估计值的标准差
    统计量["纯PSM的标准差"] = psm_arr.std(ddof=1)
    统计量["PSM-DID的标准差"] = psmdid_arr.std(ddof=1)
    统计量["纯DID的标准差"] = did_arr.std(ddof=1)
    # 新增控制协变量DID的统计量
    统计量["控制协变量DID估计值均值"] = did_cov_arr.mean()
    统计量["控制协变量DID的Bias"] = 统计量["控制协变量DID估计值均值"] - 3
    统计量["控制协变量DID的RMSE"] = np.sqrt(np.mean((did_cov_arr - 3) ** 2))
    统计量["控制协变量DID的标准差"] = did_cov_arr.std(ddof=1)
    return 统计量

def print_stats_table(stats_dict):
    """
    打印统计量表格，便于对比四种方法
    stats_dict: more_analyze 返回的字典
    """
    # 提取四种方法的名称和对应指标
    methods = ['纯PSM', 'PSM-DID', '纯DID', '控制协变量DID']
    # 需要提取的指标前缀
    metrics = ['估计值均值', 'Bias', 'RMSE', '标准差']

    # 打印表头
    print("\n{:<16} {:>12} {:>12} {:>12} {:>12}".format("方法", *metrics))
    print("-" * 68)

    # 逐行输出
    for method in methods:
        row = [method]
        for metric in metrics:
            key = f"{method}{'的' + metric if metric != '估计值均值' else metric}"  # 注意键名的组成
            # 因为键名如 "纯PSM估计值均值", "纯PSM的Bias", "纯PSM的RMSE", "纯PSM的标准差"
            # 统一格式：方法 + 指标（Bias/RMSE/标准差前加“的”，均值不加）
            if metric == '估计值均值':
                key = f"{method}{metric}"
            else:
                key = f"{method}的{metric}"
            value = stats_dict.get(key, np.nan)
            row.append(f"{value:.4f}")
        print("{:<16} {:>12} {:>12} {:>12} {:>12}".format(*row))
    print()

def esti_histogram(analyze_result,scene_name):
    '''画各个方法的估计值的分布直方图'''
    psm_arr = np.array(analyze_result[0])
    psmdid_arr = np.array(analyze_result[1])
    did_arr = np.array(analyze_result[2])
    did_cov_arr = np.array(analyze_result[3])
    true_tau = 3

    plt.figure(figsize=(16, 4))

    # 子图1: PSM
    plt.subplot(1, 4, 1)
    plt.hist(psm_arr, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(true_tau, color='red', linestyle='--', linewidth=2, label='真实值')
    plt.title(f'场景{scene_name} - 纯PSM')
    plt.xlabel('估计值')
    plt.ylabel('频数')
    plt.legend()

    # 子图2: PSM-DID
    plt.subplot(1, 4, 2)
    plt.hist(psmdid_arr, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(true_tau, color='red', linestyle='--', linewidth=2, label='真实值')
    plt.title(f'场景{scene_name} - PSM-DID')
    plt.xlabel('估计值')
    plt.ylabel('频数')
    plt.legend()

    # 子图3: DID
    plt.subplot(1, 4, 3)
    plt.hist(did_arr, bins=20, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(true_tau, color='red', linestyle='--', linewidth=2, label='真实值')
    plt.title(f'场景{scene_name} - 纯DID')
    plt.xlabel('估计值')
    plt.ylabel('频数')
    plt.legend()

    # 子图4: 控制协变量DID
    plt.subplot(1, 4, 4)
    plt.hist(did_cov_arr, bins=20, alpha=0.7, color='purple', edgecolor='black')
    plt.axvline(true_tau, color='red', linestyle='--', linewidth=2, label='真实值')
    plt.title(f'场景{scene_name} - 控制协变量DID')
    plt.xlabel('估计值')
    plt.ylabel('频数')
    plt.legend()

    plt.tight_layout()
    plt.show()

def check_balance(df_forPSM,pairs):
    '''对某次数据进行平衡性检验，输出两个SMD，第一个是匹配前的，另一个是匹配后的'''
    #匹配前SMD
    treat_before=df_forPSM[df_forPSM["Di"]==1]["Xi"]
    controll_before=df_forPSM[df_forPSM["Di"]==0]["Xi"]
    mean_t_before=np.mean(treat_before)
    mean_c_before=np.mean(controll_before)
    var_t_before=treat_before.var(ddof=1)
    var_c_before=controll_before.var(ddof=1)
    smd_before=(mean_t_before-mean_c_before)/(((var_c_before+var_t_before)/2)**(1/2))
    #匹配后SMD
    t_id=[x[0] for x in pairs]
    c_id=[x[1] for x in pairs]
    treat_after=df_forPSM[df_forPSM["id"].isin(t_id)]["Xi"]
    controll_after=df_forPSM[df_forPSM["id"].isin(c_id)]["Xi"]
    mean_t_after = np.mean(treat_after)
    mean_c_after = np.mean(controll_after)
    var_t_after = treat_after.var(ddof=1)
    var_c_after = controll_after.var(ddof=1)
    smd_after = (mean_t_after - mean_c_after) / (((var_c_after + var_t_after) / 2) ** (1 / 2))
    return [smd_before,smd_after]

def plot_common_support(df_with_pscore, scene_name):
    """
    画倾向得分共同支持图（处理组 vs 对照组）
    df_with_pscore: 已经包含 'pScore' 和 'Di' 列的 DataFrame
    scene_name: 场景名称，用于标题
    """
    treat_pscore = df_with_pscore[df_with_pscore['Di'] == 1]['pScore']
    control_pscore = df_with_pscore[df_with_pscore['Di'] == 0]['pScore']

    plt.figure(figsize=(8, 5))
    # 画直方图，alpha 设置透明度，bins 数量可调
    plt.hist(treat_pscore, bins=30, alpha=0.5, color='blue', label='处理组', density=True)
    plt.hist(control_pscore, bins=30, alpha=0.5, color='orange', label='对照组', density=True)
    plt.xlabel('倾向得分')
    plt.ylabel('密度')
    plt.title(f'场景{scene_name} - 倾向得分共同支持')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


#作业主程序
def Main():
    '''作业主程序，依次汇报场景ABC的数据分析结果,大改5,6分钟才能跑出来，敬请耐心等待'''
    print("-----------场景A分析结果------------")
    r0=analyze(500,"A",500)
    #print("500次模拟中，估计值的一些统计量：\n",more_analyze(r0))
    print("500次模拟中，估计值的一些统计量：")
    print_stats_table(more_analyze(r0))
    esti_histogram(r0,"A")

    r0=Get_A_paneldata(500)[1]
    add_core(r0)
    plot_common_support(r0,"A")
    p0=get_pairs(r0)
    print("\n")
    print("以某一次模拟为例演示“匹配前后协变量平衡情况”和“倾向得分共同支持图”")
    print("匹配后剩下的样本数",2*len(p0))
    print("匹配前后的SMD对比",check_balance(r0,p0))


    print("-----------场景B分析结果------------")
    r1=analyze(500,"B",500)
    #print("500次模拟中，估计值的一些统计量：\n",more_analyze(r1))
    print("500次模拟中，估计值的一些统计量：")
    print_stats_table(more_analyze(r1))
    esti_histogram(r1,"B")

    r1=Get_B_paneldata(500)[1]
    add_core(r1)
    plot_common_support(r1,"B")
    p1=get_pairs(r1)
    print("\n")
    print("以某一次模拟为例演示“匹配前后协变量平衡情况”和“倾向得分共同支持图”")
    print("匹配后剩下的样本数",2*len(p1))
    print("匹配前后的SMD对比",check_balance(r1,p1))


    print("-----------场景C分析结果------------")
    r2=analyze(500,"C",500)
    #print("500次模拟中，估计值的一些统计量：\n",more_analyze(r2))
    print("500次模拟中，估计值的一些统计量：")
    print_stats_table(more_analyze(r2))
    esti_histogram(r2,"C")

    r2=Get_C_paneldata(500)[1]
    add_core(r2)
    plot_common_support(r2,"C")
    p2=get_pairs(r2)
    print("\n")
    print("以某一次模拟为例演示“匹配前后协变量平衡情况”和“倾向得分共同支持图”")
    print("匹配后剩下的样本数",2*len(p2))
    print("匹配前后的SMD对比",check_balance(r2,p2))


    print("-----------场景D分析结果------------")
    r=analyze(500,"D",500)
    print("500次模拟中，估计值的一些统计量：")
    print_stats_table(more_analyze(r))
    esti_histogram(r,"D")

    r=Get_D_paneldata(500)[1]
    add_core(r)
    plot_common_support(r,"D")
    p=get_pairs(r)
    print("\n")
    print("以某一次模拟为例演示“匹配前后协变量平衡情况”和“倾向得分共同支持图”")
    print("匹配后剩下的样本数",2*len(p))
    print("匹配前后的SMD对比",check_balance(r,p))

Main()










