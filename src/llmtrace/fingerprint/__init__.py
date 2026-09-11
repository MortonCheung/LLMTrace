"""Model Identity Evidence / Fingerprint Foundation (v0.6).

与能力评测（Capability / Calibration / ReferenceSet）**完全分离**的一层：
它只回答"观测到的行为分布与哪个参考身份更接近"，不回答"模型有多强"，
也不构成上游身份的密码学证明。

本 package 自体不发送任何 HTTP 请求：所有真实请求都经由现有 Provider
（``BaseProvider.complete``），由 Provider 统一消费 RequestBudget 并记录
HTTPEvidence（Rule 4）。
"""
