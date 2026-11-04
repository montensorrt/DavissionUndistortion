# Git 命令参考手册

**项目**: Undistortion Standalone
**文档目的**: Git操作指南和命令参考
**创建日期**: 2025-11-13
**版本**: v1.0

---

## 📊 目录

1. [基础概念](#基础概念)
2. [本地仓库操作](#本地仓库操作)
3. [远程仓库操作](#远程仓库操作)
4. [分支管理](#分支管理)
5. [提交历史管理](#提交历史管理)
6. [常用工作流程](#常用工作流程)
7. [问题排查](#问题排查)

---

## 🏗️ 基础概念

### Git 工作区域

```
工作目录 (Working Directory)
    ↓ git add
暂存区 (Staging Area / Index)
    ↓ git commit
本地仓库 (Local Repository)
    ↓ git push
远程仓库 (Remote Repository)
```

### 仓库状态概念

| 状态 | 含义 | 命令查看 |
|------|------|----------|
| **Untracked** | 新文件，Git未跟踪 | `git status` |
| **Modified** | 已修改，未暂存 | `git status` |
| **Staged** | 已修改，已暂存 | `git status` |
| **Committed** | 已提交到本地仓库 | `git log` |
| **Pushed** | 已推送到远程仓库 | `git status` |

---

## 💻 本地仓库操作

### 基础状态查询

```bash
# 查看当前仓库状态
git status

# 查看工作目录和暂存区的差异
git diff

# 查看暂存区和最新提交的差异
git diff --staged

# 查看提交历史
git log
git log --oneline                    # 简洁格式
git log --graph --oneline            # 图形化格式
git log --since="2025-11-13"         # 查看指定日期后的提交
```

### 文件操作

```bash
# 添加文件到暂存区
git add <file>                       # 添加单个文件
git add .                            # 添加当前目录所有文件
git add *.cpp                        # 添加所有cpp文件
git add src/                         # 添加src目录

# 从暂存区移除文件
git reset <file>                     # 从暂存区移除，保留工作目录修改
git reset -- <file>                  # 同上（推荐使用）

# 丢弃工作目录的修改
git checkout -- <file>               # 丢弃单个文件的修改
git checkout .                       # 丢弃所有文件的修改

# 删除文件
git rm <file>                        # 删除文件并暂存
git rm --cached <file>               # 从暂存区删除，保留工作目录文件
```

### 本地提交操作 ⭐ **【仅本地，不影响远程】**

```bash
# 创建提交（仅本地）
git commit -m "提交信息"

# 创建详细提交
git commit -m "feat: 添加新功能

- 详细描述1
- 详细描述2"

# 交互式提交（可以逐个文件选择）
git commit -i

# 修改最后一次提交
git commit --amend                   # 修改最后一次提交信息
git commit --amend -m "新的提交信息"

# 提交时跳过暂存区（直接提交所有修改）
git commit -am "提交信息"            # 仅限已跟踪文件的修改
```

---

## 🌐 远程仓库操作

### 远程仓库管理

```bash
# 查看远程仓库
git remote -v
git remote show origin               # 查看origin详细信息

# 添加远程仓库
git remote add <name> <url>
git remote add origin https://github.com/user/repo.git

# 删除远程仓库
git remote remove <name>
git remote rm origin
```

### 获取远程更新 ⭐ **【从远程拉取，不影响本地提交】**

```bash
# 获取远程更新，但不合并到本地分支
git fetch origin
git fetch origin master              # 获取远程master分支更新

# 获取远程更新并合并到本地分支（可能会产生冲突）
git pull origin master
git pull                            # 拉取当前分支的远程更新

# 获取远程更新并变基（保持线性历史）
git pull --rebase origin master
```

### 推送到远程 ⭐ **【将本地提交推送到远程】**

```bash
# 推送本地提交到远程
git push origin master
git push                            # 推送当前分支到远程

# 强制推送（谨慎使用）
git push --force origin master       # 强制推送（覆盖远程）
git push --force-with-lease origin master  # 相对安全的强制推送

# 首次推送新分支
git push -u origin feature-branch

# 推送所有分支
git push --all origin
```

### 同步操作总结

| 操作 | 本地影响 | 远程影响 | 用途 |
|------|----------|----------|------|
| `git commit` | ✅ 创建提交 | ❌ 无影响 | 保存本地修改 |
| `git push` | ❌ 无影响 | ✅ 上传提交 | 共享本地提交 |
| `git pull` | ✅ 合并更新 | ❌ 无影响 | 获取他人修改 |
| `git fetch` | ❌ 仅获取 | ❌ 无影响 | 查看远程更新 |

---

## 🌿 分支管理

### 分支基础操作

```bash
# 查看分支
git branch                          # 查看本地分支
git branch -r                       # 查看远程分支
git branch -a                       # 查看所有分支

# 创建分支
git branch <branch-name>            # 创建但不切换
git checkout -b <branch-name>       # 创建并切换

# 切换分支
git checkout <branch-name>
git switch <branch-name>            # Git 2.23+ 新命令

# 删除分支
git branch -d <branch-name>         # 删除已合并的分支
git branch -D <branch-name>         # 强制删除分支
```

### 分支合并操作

```bash
# 合并分支到当前分支
git merge <source-branch>

# 变基（重新基础）
git rebase <base-branch>
git rebase master                   # 将当前分支变基到master

# 中断合并/变基
git merge --abort                   # 中断合并
git rebase --abort                  # 中断变基

# 继续合并/变基（解决冲突后）
git merge --continue
git rebase --continue
```

### 推送分支

```bash
# 推送新分支到远程
git push -u origin <branch-name>

# 推送所有分支
git push --all origin

# 删除远程分支
git push origin --delete <branch-name>
git push origin :<branch-name>     # 同上，另一种语法
```

---

## 📜 提交历史管理

### 查看历史

```bash
# 基本查看
git log
git log --oneline                   # 简洁格式
git log --graph --oneline           # 图形化格式
git log --stat                      # 显示修改统计

# 格式化查看
git log --pretty=format:"%h %s %an %ar"

# 查看特定文件历史
git log <file>
git log -p <file>                   # 显示具体修改内容

# 查看指定范围提交
git log HEAD~5..HEAD                # 最近5个提交
git log since="2025-11-13"         # 指定日期后
```

### 历史操作

```bash
# 回退到指定提交
git reset --hard <commit-hash>      # 完全回退，丢弃所有修改
git reset --soft <commit-hash>      # 软回退，保留修改在暂存区
git reset <commit-hash>             # 混合回退，保留修改在工作目录

# 撤销某个提交（创建新提交来撤销）
git revert <commit-hash>

# 挑选提交（将特定提交应用到当前分支）
git cherry-pick <commit-hash>

# 交互式重写历史（谨慎使用）
git rebase -i HEAD~5                # 交互式编辑最近5个提交
```

---

## 🔄 常用工作流程

### 日常开发工作流

```bash
# 1. 开始工作前，确保本地最新
git pull origin master

# 2. 创建功能分支
git checkout -b feature/new-feature

# 3. 开发过程中定期提交
git add .
git commit -m "feat: 实现新功能的一部分"

# 4. 完成功能，推送到远程
git push -u origin feature/new-feature

# 5. 合并到主分支
git checkout master
git pull origin master
git merge feature/new-feature
git push origin master

# 6. 清理
git branch -d feature/new-feature
```

### 紧急修复工作流

```bash
# 1. 从主分支创建修复分支
git checkout master
git checkout -b hotfix/urgent-fix

# 2. 快速修复并提交
git add .
git commit -m "fix: 修复紧急问题"

# 3. 直接合并到主分支和开发分支
git checkout master
git merge hotfix/urgent-fix
git push origin master

git checkout develop
git merge hotfix/urgent-fix
git push origin develop

# 4. 清理
git branch -d hotfix/urgent-fix
```

### 本地实验工作流

```bash
# 1. 创建实验分支
git checkout -b experiment/try-something

# 2. 大胆实验，多次提交
git add .
git commit -m "实验: 尝试方案A"
git add .
git commit -m "实验: 方案A失败，尝试方案B"

# 3. 实验成功，整理历史
git rebase -i HEAD~3                # 整理提交历史

# 4. 合并到主分支
git checkout master
git merge --squash experiment/try-something  # 压缩提交
git commit -m "feat: 添加实验成功的功能"

# 5. 删除实验分支
git branch -D experiment/try-something
```

---

## 🔧 问题排查

### 常见问题解决

#### 合并冲突

```bash
# 1. 查看冲突文件
git status

# 2. 手动编辑冲突文件
# <<< HEAD
# 当前分支的内容
# =====
# 要合并分支的内容
# >>>> other-branch

# 3. 解决冲突后标记为已解决
git add <conflicted-file>

# 4. 继续合并
git merge --continue

# 5. 或者取消合并
git merge --abort
```

#### 推送被拒绝

```bash
# 问题：远程有本地没有的提交
git push origin master
# ! [rejected] master -> master (non-fast-forward)

# 解决方案1：先拉取再推送
git pull origin master
git push origin master

# 解决方案2：变基后推送
git pull --rebase origin master
git push origin master
```

#### 撤销错误的提交

```bash
# 情况1：还未推送到远程
git reset --hard HEAD~1              # 撤销最后一次提交
git reset --hard <commit-hash>       # 撤销到指定提交

# 情况2：已经推送到远程
git revert HEAD                      # 创建新提交来撤销
git revert <commit-hash>             # 撤销指定提交
git push origin master
```

#### 误删文件恢复

```bash
# 情况1：文件在暂存区
git checkout HEAD -- <file>

# 情况2：文件已提交但被删除
git checkout <commit-hash> -- <file>

# 查看文件历史
git log -- <file>
```

### 配置问题

```bash
# 查看配置
git config --list

# 配置用户信息
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 配置编辑器
git config --global core.editor "code --wait"  # VSCode

# 配置差异工具
git config --global diff.tool vscode
git config --global difftool.vscode.cmd 'code --wait --diff $LOCAL $REMOTE'
```

---

## 📝 提交信息规范

### 提交信息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 类型说明

| 类型 | 说明 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat: 添加LUT优化功能` |
| `fix` | 修复bug | `fix: 修复内存泄漏问题` |
| `docs` | 文档更新 | `docs: 更新API文档` |
| `style` | 代码格式 | `style: 修复代码格式` |
| `refactor` | 重构 | `refactor: 重构权重加载逻辑` |
| `test` | 测试 | `test: 添加单元测试` |
| `chore` | 构建工具或辅助工具的变动 | `chore: 更新构建脚本` |

### 示例

```bash
git commit -m "feat(performance): 添加权重量化优化

- 将float权重转换为int8量化权重
- 使用SIMD整数指令提升性能
- 添加反量化步骤保证精度

Closes #123"
```

---

## ⚡ 快速参考

### 常用命令速查

```bash
# 本地操作
git status                           # 查看状态
git add .                           # 暂存所有修改
git commit -m "msg"                 # 提交到本地
git log --oneline                   # 查看提交历史

# 远程操作
git pull                            # 拉取远程更新
git push                            # 推送本地提交
git fetch                           # 获取远程信息

# 分支操作
git branch                          # 查看分支
git checkout -b new-branch          # 创建新分支
git merge feature-branch            # 合并分支
```

### 紧急命令

```bash
# 撤销最后提交（未推送）
git reset --hard HEAD~1

# 撤销工作目录修改
git checkout -- .

# 查看最近的修改
git diff HEAD~1

# 暂存当前工作
git stash
git stash pop
```

---

**文档维护**: 随时更新
**下次更新**: 2025-11-20

---

*此文档提供常用Git命令的参考指南，建议收藏使用。*