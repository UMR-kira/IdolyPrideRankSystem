import email
import os
import re
import shutil
import pandas as pd
from email import policy
from urllib.parse import unquote, urlparse
from bs4 import BeautifulSoup

"""
首先打开wiki官网排行榜然后右键另存为mhtml文件再使用程序处理
https://wiki.biligame.com/idolypride/卡牌排行
"""

"""解析MHTML文件，返回HTML内容、资源映射和资源保存路径映射"""
def parse_mhtml(mhtml_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    resource_dir = os.path.join(output_dir, "imgcache")
    os.makedirs(resource_dir, exist_ok=True)
    with open(mhtml_path, 'rb') as f:
        msg = email.message_from_binary_file(f, policy=policy.default)
    html_content = None
    resource_map = {}
    base_url = ""
    # 遍历所有MIME部分
    for part in msg.walk():
        content_type = part.get_content_type()
        content_location = part.get('Content-Location', '')
        content_id = part.get('Content-ID', '').strip('<>')
        # 记录HTML页面的基础URL
        if content_type == 'text/html' and not base_url:
            base_url = content_location
        # 处理HTML主体
        if content_type == 'text/html':
            payload = part.get_payload(decode=True)
            charset = part.get_content_charset() or 'utf-8'
            try:
                html_content = payload.decode(charset)
            except UnicodeDecodeError:
                html_content = payload.decode('latin1', errors='ignore')
        # 处理资源文件 (图片等)
        elif content_type.startswith('image/'):
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            # 生成唯一文件名
            if content_location:
                # 从URL中提取文件名
                filename = os.path.basename(urlparse(content_location).path)
                # 保存文件
                save_path = os.path.join(resource_dir, filename)
                # 检查文件是否已存在
                if not os.path.exists(save_path):
                    # 写入文件
                    with open(save_path, 'wb') as f:
                        f.write(payload)
                # 记录保存路径
                if content_location:
                    resource_map[content_location] = save_path
                if content_id:
                    resource_map[f"cid:{content_id}"] = save_path
    if not html_content:
        raise ValueError("HTML content not found in MHTML")
    return html_content, resource_map, base_url

"""根据内容类型获取文件扩展名"""
def get_extension(content_type):
    mapping = {
        'image/jpeg': '.jpg',
        'image/png': '.png',
        'image/gif': '.gif',
        'image/webp': '.webp'
    }
    return mapping.get(content_type, '.bin')

"""解析MHTML获取HTML内容和资源"""
def extract_data(mhtml_path, output_dir):
    html_content, resource_map, base_url= parse_mhtml(mhtml_path, output_dir)
    soup = BeautifulSoup(html_content, 'html.parser')
    card_data = []
    # 定位所有卡牌表格 (修正后的选择器)
    tables = soup.find_all('table', class_="wikitable")
    for table in tables:
        # 跳过移动端表格
        if table.find_parent(class_="visible-xs"):
            continue
        # 获取分类标题
        h2_tag = table.find_previous('h2')
        category = h2_tag.contents[1].text if h2_tag else "Unknown"
        category = category.strip()
        # 获取表头类别
        headers = []
        for th in table.select('tr:first-child th')[1:]:
            header_text = th.get_text(strip=True)
            headers.append(unquote(header_text.replace('=', '%')))
        # 处理表格行
        colormap = {'#FFF0F5': '歌唱-红轨', '#E0FFFF': '舞蹈-蓝轨', '#FFFFE0': '表演-黄轨'}
        for row in table.select('tr')[1:]:  # 跳过表头
            cells = row.select('td')
            if len(cells) < 2:
                continue
            # 提取强度等级
            strength = cells[0].get_text(strip=True)
            # 处理每个类别的单元格
            for idx, cell in enumerate(cells[1:]):
                try:
                    header = headers[idx]
                except:
                    header = headers[0] # 附表只有单类
                # 提取所有卡牌链接
                if strength in ['特殊', '辅助sp', 'CT↓'] and len(headers) == 1:
                    cell = cell.contents[0]
                for content in cell.contents:
                    try:
                        img_tag = content.find_all('img')
                    except AttributeError:
                        continue
                    if not img_tag:
                        continue
                    # 获取图片信息
                    try:
                        name = img_tag[0].get('alt')
                        card_name = name[0:-11]+'.png'
                        img_src = img_tag[0].attrs.get('src', '')
                        idol_type = str(img_tag[1].attrs.get('alt', '')).split('.')[0].split('-')[-1]
                        idol_rarity = str(img_tag[2].attrs.get('alt', '')).split('.')[0].split('-')[-1]
                    except AttributeError:
                        continue
                    try:
                        if cell.get('style', ''):
                            backcolor = cell['style']
                            match = re.search(r'(#FFF0F5|#E0FFFF|#FFFFE0)', backcolor)
                            if match:
                                color = str(match.group(1))
                                railcolor = colormap.get(color)
                            else:
                                railcolor = '无限制'
                    except AttributeError:
                        print("轨道色提取失败")

                    if img_src in resource_map:
                        # 处理资源映射中的图片
                        if img_src in resource_map:
                            img_path = resource_map[img_src]  # 源图片路径
                            card_dir = os.path.join(output_dir, "card")
                            os.makedirs(card_dir, exist_ok=True)
                            # 构建保存路径
                            save_path = os.path.join(card_dir, card_name)
                            # 复制文件
                            if not os.path.exists(save_path):
                                shutil.copy2(img_path, save_path)
                            card_path = save_path

                    # 添加到数据列表
                    if img_path:
                        card_data.append({
                            'category': category,
                            'strength': strength,
                            'tableheader': header,
                            'railcolor': railcolor,
                            'card_name': card_name,
                            'idol_type': idol_type,
                            'idol_rarity': idol_rarity,
                            'card_path': card_path
                        })
    return card_data

"""将卡牌数据保存到Excel文件"""
def save_excel(card_data, output_path):
    df = pd.DataFrame(card_data)
    df = df[['category', 'strength', 'tableheader', 'railcolor', 'card_name', 'idol_type', 'idol_rarity', 'card_path']]
    df.to_excel(output_path, index=False)
    print(f"卡牌数据已保存到: {output_path}")

"""提取主要的卡牌排行数据,合并多张卡"""
def merge_card_ranks(df):
    # 定义主要排行类型
    main_ranks = ['通常排行', '对决排行']
    # 获取所有基础列（除category和strength外的列）
    base_columns = [col for col in df.columns if col not in ['category', 'strength', 'railcolor']]
    card_dict = {}
    for _, row in df.iterrows():
        card_name = row['card_name']
        category = row['category']
        strength = row['strength']
        railcolor = row['railcolor']
        # 检查是否已存在该卡牌记录
        if card_name not in card_dict:
            # 如果是新卡牌，初始化记录
            card_dict[card_name] = {
                'base_info': {col: row[col] for col in base_columns},
                'main_ranks': {rank: None for rank in main_ranks},
                'other_ranks': {},  # 初始化其他排行字典
                'railcolor': '无限制'
            }
        # 更新排行信息（无论是否是新卡牌）
        card_dict[card_name]['railcolor'] = railcolor
        if category in main_ranks:
            card_dict[card_name]['main_ranks'][category] = strength
        else:
            # 添加其他排行信息
            card_dict[card_name]['other_ranks'][category] = strength
    # 构建结果DataFrame
    results = []
    for card, data in card_dict.items():
        # 构建主要排行文本 - 只显示有实际值的排行
        main_items = []
        for rank in main_ranks:
            value = data['main_ranks'][rank]
            if value:  # 只有当值存在时才添加
                main_items.append(f"{rank}:{value}")

        main_str = ",".join(main_items) if main_items else ""

        # 构建其他排行文本（过滤空值）
        other_items = [f"{k}:{v}" for k, v in data['other_ranks'].items() if v is not None]
        other_str = ",".join(other_items) if other_items else ""

        # 创建结果行：基础信息 + 两个排行列
        result_row = data['base_info'].copy()  # 复制基础信息
        result_row.update({
            'card_name': card,
            'main_ranks': main_str,
            'other_ranks': other_str,
            'railcolor': data['railcolor']
        })
        results.append(result_row)
    return pd.DataFrame(results)

if __name__ == "__main__":
    mhtml_path = "网页文件地址.mhtml"
    output_dir = "导出数据的文件夹"
    # 提取卡牌数据
    card_data = extract_data(mhtml_path, output_dir)
    print(f"成功提取 {len(card_data)} 条卡牌数据")
    excel_path = output_dir + "\CardData.xlsx"
    save_excel(card_data, excel_path)
    # 提取排行数据
    input_path = output_dir + "\CardData.xlsx"
    df = pd.read_excel(input_path)
    result_df = merge_card_ranks(df)
    output_path = output_dir + "\CardRank.xlsx"
    result_df.to_excel(output_path, index=False)
    print("卡牌排行数据保存成功")