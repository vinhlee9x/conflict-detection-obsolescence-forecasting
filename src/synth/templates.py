"""Vietnamese document templates for the two synthetic tenants.

Each clause has several phrasings that carry the same slots, so a paraphrase can
reword a clause without changing any value, and a conflict can change a value
without rewording.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Slot:
    choices: tuple


@dataclass(frozen=True)
class Clause:
    key: str
    phrasings: tuple
    slots: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Topic:
    key: str
    title: str
    department: str
    clauses: tuple


@dataclass(frozen=True)
class Tenant:
    key: str
    name: str
    code_prefix: str
    departments: tuple
    subjects: tuple
    topics: tuple
    boilerplate: tuple


def _c(key, phrasings, **slots):
    return Clause(key=key, phrasings=tuple(phrasings), slots={k: Slot(tuple(v)) for k, v in slots.items()})


TECH_TOPICS = (
    Topic(
        "backup",
        "Quy trình sao lưu dữ liệu",
        "Hạ tầng",
        (
            _c("freq", [
                "Dữ liệu của {subject} được sao lưu toàn bộ mỗi {hours} giờ một lần.",
                "Hệ thống {subject} phải thực hiện bản sao lưu đầy đủ với chu kỳ {hours} giờ.",
            ], hours=[6, 12, 24, 48]),
            _c("retention", [
                "Bản sao lưu được lưu giữ tối thiểu {days} ngày trước khi xoá.",
                "Thời gian giữ bản sao lưu không được ngắn hơn {days} ngày.",
            ], days=[14, 30, 60, 90]),
            _c("offsite", [
                "Ít nhất {copies} bản sao phải được lưu tại trung tâm dữ liệu dự phòng.",
                "Trung tâm dữ liệu dự phòng giữ tối thiểu {copies} bản sao của mỗi kỳ sao lưu.",
            ], copies=[1, 2, 3]),
            _c("restore_test", [
                "Việc khôi phục thử từ bản sao lưu được kiểm tra mỗi {months} tháng.",
                "Đội vận hành chạy diễn tập khôi phục dữ liệu với chu kỳ {months} tháng.",
            ], months=[1, 3, 6]),
            _c("approver", [
                "Mọi yêu cầu xoá bản sao lưu phải được {role} phê duyệt.",
                "Chỉ {role} có quyền phê duyệt việc xoá bản sao lưu.",
            ], role=["trưởng phòng Hạ tầng", "giám đốc Công nghệ", "quản trị viên hệ thống"]),
            _c("encryption", [
                "Bản sao lưu được mã hoá bằng khoá có độ dài {bits} bit.",
                "Khoá mã hoá dùng cho bản sao lưu có độ dài tối thiểu {bits} bit.",
            ], bits=[128, 192, 256]),
        ),
    ),
    Topic(
        "deploy",
        "Quy trình triển khai phiên bản mới",
        "Phát triển phần mềm",
        (
            _c("window", [
                "Phiên bản mới của {subject} chỉ được triển khai trong khung giờ từ {start} giờ đến {end} giờ.",
                "Khung giờ triển khai cho {subject} bắt đầu lúc {start} giờ và kết thúc lúc {end} giờ.",
            ], start=[20, 21, 22, 23], end=[2, 4, 5, 6]),
            _c("review", [
                "Mỗi thay đổi cần ít nhất {reviewers} người duyệt mã trước khi hợp nhất.",
                "Trước khi hợp nhất, thay đổi phải có tối thiểu {reviewers} lượt duyệt mã.",
            ], reviewers=[1, 2, 3]),
            _c("staging", [
                "Bản dựng phải chạy ổn định trên môi trường thử nghiệm ít nhất {days} ngày.",
                "Thời gian chạy thử trên môi trường thử nghiệm tối thiểu là {days} ngày.",
            ], days=[1, 2, 3, 5, 7]),
            _c("rollback", [
                "Nếu tỉ lệ lỗi vượt {percent}% trong 30 phút đầu, phải hoàn tác ngay phiên bản.",
                "Phiên bản bị hoàn tác khi tỉ lệ lỗi sau triển khai lớn hơn {percent}%.",
            ], percent=[1, 2, 5, 10]),
            _c("approver", [
                "Việc triển khai lên môi trường thật cần {role} phê duyệt.",
                "{role} là người phê duyệt cuối cùng cho mỗi lần triển khai lên môi trường thật.",
            ], role=["trưởng nhóm phát triển", "quản lý sản phẩm", "giám đốc Kỹ thuật"]),
            _c("freeze", [
                "Không triển khai trong {days} ngày trước các đợt cao điểm kinh doanh.",
                "Thời gian đóng băng triển khai trước đợt cao điểm là {days} ngày.",
            ], days=[3, 5, 7, 14]),
        ),
    ),
    Topic(
        "incident",
        "Quy trình xử lý sự cố",
        "Vận hành hệ thống",
        (
            _c("ack", [
                "Sự cố mức nghiêm trọng của {subject} phải được tiếp nhận trong vòng {minutes} phút.",
                "Thời gian tiếp nhận tối đa cho sự cố nghiêm trọng trên {subject} là {minutes} phút.",
            ], minutes=[5, 10, 15, 30]),
            _c("resolve", [
                "Sự cố nghiêm trọng cần được khắc phục trong vòng {hours} giờ.",
                "Thời hạn khắc phục sự cố nghiêm trọng là {hours} giờ kể từ lúc tiếp nhận.",
            ], hours=[2, 4, 8, 24]),
            _c("escalate", [
                "Nếu chưa khắc phục sau {hours} giờ, sự cố được chuyển lên {role}.",
                "Sự cố quá {hours} giờ chưa xử lý xong sẽ được báo cáo lên {role}.",
            ], hours=[1, 2, 4], role=["trưởng ca vận hành", "giám đốc Công nghệ", "trưởng phòng Vận hành"]),
            _c("postmortem", [
                "Báo cáo phân tích nguyên nhân gốc phải hoàn thành trong {days} ngày làm việc.",
                "Trong vòng {days} ngày làm việc, đội vận hành gửi báo cáo nguyên nhân gốc.",
            ], days=[2, 3, 5, 7]),
            _c("notify", [
                "Khách hàng bị ảnh hưởng được thông báo trong vòng {minutes} phút.",
                "Thông báo tới khách hàng bị ảnh hưởng không chậm hơn {minutes} phút.",
            ], minutes=[15, 30, 60]),
        ),
    ),
    Topic(
        "access",
        "Chính sách cấp quyền truy cập",
        "Bảo mật",
        (
            _c("review", [
                "Quyền truy cập vào {subject} được rà soát lại mỗi {months} tháng.",
                "Chu kỳ rà soát quyền truy cập {subject} là {months} tháng.",
            ], months=[1, 3, 6, 12]),
            _c("revoke", [
                "Quyền của nhân viên nghỉ việc phải bị thu hồi trong vòng {hours} giờ.",
                "Thời hạn thu hồi quyền khi nhân viên nghỉ việc là {hours} giờ.",
            ], hours=[4, 8, 24, 48]),
            _c("password", [
                "Mật khẩu phải có tối thiểu {chars} ký tự.",
                "Độ dài mật khẩu không được ít hơn {chars} ký tự.",
            ], chars=[8, 10, 12, 14]),
            _c("rotation", [
                "Mật khẩu tài khoản quản trị được đổi mỗi {days} ngày.",
                "Tài khoản quản trị phải thay mật khẩu với chu kỳ {days} ngày.",
            ], days=[30, 60, 90, 180]),
            _c("approver", [
                "Quyền quản trị chỉ được cấp khi có phê duyệt của {role}.",
                "{role} phê duyệt mọi yêu cầu cấp quyền quản trị.",
            ], role=["trưởng phòng Bảo mật", "giám đốc Công nghệ", "quản lý trực tiếp"]),
            _c("lockout", [
                "Tài khoản bị khoá sau {attempts} lần đăng nhập sai liên tiếp.",
                "Sau {attempts} lần nhập sai mật khẩu, tài khoản sẽ bị khoá tạm thời.",
            ], attempts=[3, 5, 10]),
        ),
    ),
    Topic(
        "monitoring",
        "Hướng dẫn giám sát hệ thống",
        "Vận hành hệ thống",
        (
            _c("cpu", [
                "Cảnh báo được phát khi CPU của {subject} vượt {percent}% trong 5 phút.",
                "Ngưỡng cảnh báo CPU cho {subject} là {percent}% kéo dài 5 phút.",
            ], percent=[70, 80, 85, 90]),
            _c("disk", [
                "Dung lượng đĩa còn trống dưới {percent}% phải được xử lý trong ngày.",
                "Khi đĩa còn trống ít hơn {percent}%, đội vận hành xử lý ngay trong ngày.",
            ], percent=[10, 15, 20]),
            _c("log_retention", [
                "Nhật ký hệ thống được lưu trữ {days} ngày.",
                "Thời gian lưu nhật ký hệ thống là {days} ngày.",
            ], days=[30, 90, 180, 365]),
            _c("oncall", [
                "Mỗi ca trực giám sát kéo dài {hours} giờ.",
                "Ca trực giám sát có độ dài {hours} giờ.",
            ], hours=[8, 12, 24]),
            _c("uptime", [
                "Mục tiêu thời gian hoạt động của {subject} là {percent}% mỗi tháng.",
                "{subject} cam kết thời gian hoạt động {percent}% theo tháng.",
            ], percent=["99,5", "99,9", "99,95"]),
        ),
    ),
    Topic(
        "retention",
        "Chính sách lưu trữ và xoá dữ liệu",
        "Dữ liệu",
        (
            _c("customer", [
                "Dữ liệu khách hàng trong {subject} được lưu tối đa {years} năm sau khi kết thúc hợp đồng.",
                "Sau khi hợp đồng kết thúc, dữ liệu khách hàng trên {subject} chỉ được giữ {years} năm.",
            ], years=[1, 2, 3, 5]),
            _c("anonymize", [
                "Dữ liệu dùng cho phân tích phải được ẩn danh trong vòng {days} ngày.",
                "Thời hạn ẩn danh dữ liệu phục vụ phân tích là {days} ngày.",
            ], days=[7, 14, 30]),
            _c("delete_request", [
                "Yêu cầu xoá dữ liệu cá nhân được xử lý trong {days} ngày.",
                "Hệ thống hoàn tất yêu cầu xoá dữ liệu cá nhân trong vòng {days} ngày.",
            ], days=[7, 15, 30]),
            _c("approver", [
                "Việc xuất dữ liệu ra ngoài cần sự đồng ý của {role}.",
                "Mọi lần xuất dữ liệu ra ngoài phải được {role} chấp thuận.",
            ], role=["cán bộ bảo vệ dữ liệu", "trưởng phòng Dữ liệu", "giám đốc Pháp chế"]),
            _c("audit", [
                "Nhật ký truy cập dữ liệu nhạy cảm được kiểm toán mỗi {months} tháng.",
                "Chu kỳ kiểm toán nhật ký truy cập dữ liệu nhạy cảm là {months} tháng.",
            ], months=[1, 3, 6]),
        ),
    ),
)

TECH_SUBJECTS = (
    "hệ thống ERP", "hệ thống CRM", "kho dữ liệu", "cổng thanh toán", "ứng dụng di động",
    "hệ thống email", "cổng thông tin nội bộ", "hệ thống quản lý kho", "nền tảng thương mại điện tử",
    "hệ thống kế toán", "hệ thống nhân sự", "API đối tác", "hệ thống báo cáo", "dịch vụ xác thực",
    "hệ thống tổng đài", "hệ thống hoá đơn điện tử", "nền tảng dữ liệu lớn", "hệ thống chấm công",
    "hệ thống quản lý tài liệu", "hệ thống đặt hàng trực tuyến", "hệ thống quản lý dự án",
    "hệ thống giám sát camera", "cổng dịch vụ khách hàng", "hệ thống điểm bán hàng",
    "hệ thống quản lý hợp đồng", "dịch vụ lưu trữ tệp", "hệ thống quản lý tài sản",
    "nền tảng học trực tuyến", "hệ thống quản lý đội xe", "hệ thống tích điểm khách hàng",
    "cổng tuyển dụng", "hệ thống ký số", "hệ thống quản lý mã nguồn", "dịch vụ gửi tin nhắn",
)

POLICY_TOPICS = (
    Topic(
        "returns",
        "Quy trình đổi trả đơn hàng",
        "Chăm sóc khách hàng",
        (
            _c("window", [
                "Khách hàng {subject} được đổi trả đơn hàng trong vòng {days} ngày kể từ ngày nhận hàng.",
                "Thời hạn đổi trả đơn hàng cho khách hàng {subject} là {days} ngày tính từ ngày nhận.",
            ], days=[3, 5, 7, 10, 14, 30]),
            _c("condition", [
                "Sản phẩm đổi trả phải còn {condition}.",
                "Đơn hàng chỉ được chấp nhận đổi trả khi sản phẩm còn {condition}.",
            ], condition=["nguyên tem và hoá đơn", "nguyên hộp, chưa qua sử dụng", "đầy đủ phụ kiện và phiếu bảo hành"]),
            _c("refund", [
                "Tiền hoàn lại được chuyển cho khách trong {days} ngày làm việc.",
                "Quy trình hoàn tiền đơn hàng hoàn tất trong vòng {days} ngày làm việc.",
            ], days=[3, 5, 7, 10]),
            _c("approver", [
                "Đơn hàng đổi trả có giá trị trên {amount} triệu đồng phải được {role} phê duyệt.",
                "{role} phê duyệt các đơn hàng đổi trả trị giá trên {amount} triệu đồng.",
            ], amount=[2, 5, 10, 20], role=["trưởng cửa hàng", "quản lý khu vực", "trưởng phòng Chăm sóc khách hàng"]),
            _c("fee", [
                "Phí vận chuyển đổi trả do {payer} chi trả.",
                "{payer} chịu phí vận chuyển khi đổi trả đơn hàng.",
            ], payer=["công ty", "khách hàng", "đơn vị vận chuyển"]),
        ),
    ),
    Topic(
        "leave",
        "Quy định nghỉ phép",
        "Nhân sự",
        (
            _c("annual", [
                "Nhân viên {subject} được hưởng {days} ngày phép năm.",
                "Số ngày phép năm của nhân viên {subject} là {days} ngày.",
            ], days=[12, 14, 15, 18]),
            _c("notice", [
                "Đơn xin nghỉ phép phải gửi trước ít nhất {days} ngày làm việc.",
                "Nhân viên nộp đơn nghỉ phép trước tối thiểu {days} ngày làm việc.",
            ], days=[1, 3, 5, 7]),
            _c("carry", [
                "Ngày phép chưa dùng được chuyển sang năm sau tối đa {days} ngày.",
                "Tối đa {days} ngày phép còn lại được chuyển sang năm kế tiếp.",
            ], days=[0, 3, 5, 10]),
            _c("approver", [
                "Đơn nghỉ phép dài hơn {days} ngày phải được {role} phê duyệt.",
                "{role} phê duyệt các đơn nghỉ phép trên {days} ngày.",
            ], days=[3, 5, 7], role=["quản lý trực tiếp", "trưởng phòng Nhân sự", "giám đốc chi nhánh"]),
            _c("sick", [
                "Nghỉ ốm trên {days} ngày cần có giấy xác nhận của cơ sở y tế.",
                "Giấy xác nhận y tế là bắt buộc khi nghỉ ốm quá {days} ngày.",
            ], days=[1, 2, 3]),
        ),
    ),
    Topic(
        "purchase",
        "Quy trình phê duyệt đơn hàng mua sắm",
        "Mua hàng",
        (
            _c("threshold", [
                "Đơn hàng mua sắm của {subject} trên {amount} triệu đồng phải có {quotes} báo giá.",
                "Với đơn hàng mua sắm trên {amount} triệu đồng, {subject} cần thu thập {quotes} báo giá.",
            ], amount=[10, 20, 50, 100], quotes=[2, 3]),
            _c("approver", [
                "Đơn hàng mua sắm vượt hạn mức phải được {role} phê duyệt.",
                "{role} là người phê duyệt đơn hàng mua sắm vượt hạn mức.",
            ], role=["giám đốc Tài chính", "trưởng phòng Mua hàng", "tổng giám đốc"]),
            _c("payment", [
                "Nhà cung cấp được thanh toán trong vòng {days} ngày sau khi nghiệm thu.",
                "Thời hạn thanh toán cho nhà cung cấp là {days} ngày kể từ khi nghiệm thu.",
            ], days=[15, 30, 45, 60]),
            _c("advance", [
                "Tạm ứng cho nhà cung cấp không vượt quá {percent}% giá trị đơn hàng.",
                "Mức tạm ứng tối đa là {percent}% giá trị đơn hàng mua sắm.",
            ], percent=[10, 20, 30, 50]),
            _c("vendor_review", [
                "Danh sách nhà cung cấp được đánh giá lại mỗi {months} tháng.",
                "Chu kỳ đánh giá lại nhà cung cấp là {months} tháng.",
            ], months=[6, 12, 24]),
        ),
    ),
    Topic(
        "travel",
        "Quy định công tác phí",
        "Tài chính",
        (
            _c("per_diem", [
                "Phụ cấp công tác của nhân viên {subject} là {amount} nghìn đồng mỗi ngày.",
                "Nhân viên {subject} đi công tác được phụ cấp {amount} nghìn đồng một ngày.",
            ], amount=[200, 300, 400, 500]),
            _c("hotel", [
                "Chi phí khách sạn được thanh toán tối đa {amount} nghìn đồng mỗi đêm.",
                "Mức trần chi phí khách sạn là {amount} nghìn đồng một đêm.",
            ], amount=[600, 800, 1000, 1500]),
            _c("settle", [
                "Hồ sơ quyết toán công tác phí nộp trong vòng {days} ngày sau chuyến đi.",
                "Nhân viên quyết toán công tác phí chậm nhất {days} ngày sau khi về.",
            ], days=[5, 7, 10, 15]),
            _c("approver", [
                "Chuyến công tác nước ngoài phải được {role} phê duyệt.",
                "{role} phê duyệt các chuyến công tác nước ngoài.",
            ], role=["giám đốc chi nhánh", "giám đốc Tài chính", "tổng giám đốc"]),
            _c("flight", [
                "Vé máy bay hạng thương gia chỉ áp dụng cho chặng bay trên {hours} giờ.",
                "Chặng bay dài hơn {hours} giờ mới được mua vé hạng thương gia.",
            ], hours=[3, 5, 8]),
        ),
    ),
    Topic(
        "warranty",
        "Chính sách bảo hành sản phẩm",
        "Chăm sóc khách hàng",
        (
            _c("period", [
                "Sản phẩm bán tại {subject} được bảo hành {months} tháng.",
                "Thời gian bảo hành sản phẩm mua ở {subject} là {months} tháng.",
            ], months=[6, 12, 18, 24]),
            _c("turnaround", [
                "Sản phẩm bảo hành được trả lại khách trong {days} ngày làm việc.",
                "Thời gian xử lý bảo hành không quá {days} ngày làm việc.",
            ], days=[3, 5, 7, 14]),
            _c("replace", [
                "Sản phẩm lỗi được đổi mới nếu hỏng trong {days} ngày đầu.",
                "Trong {days} ngày đầu sử dụng, sản phẩm lỗi được đổi sản phẩm mới.",
            ], days=[7, 15, 30]),
            _c("approver", [
                "Yêu cầu bảo hành ngoài chính sách cần {role} phê duyệt.",
                "{role} phê duyệt các trường hợp bảo hành ngoài chính sách.",
            ], role=["trưởng trung tâm bảo hành", "quản lý khu vực", "trưởng phòng Chăm sóc khách hàng"]),
            _c("proof", [
                "Khách hàng cần xuất trình {proof} khi yêu cầu bảo hành.",
                "Yêu cầu bảo hành phải kèm {proof}.",
            ], proof=["hoá đơn mua hàng", "phiếu bảo hành", "mã đơn hàng"]),
        ),
    ),
    Topic(
        "delivery",
        "Quy trình giao nhận đơn hàng",
        "Kho vận",
        (
            _c("dispatch", [
                "Đơn hàng của {subject} được xuất kho trong vòng {hours} giờ sau khi xác nhận.",
                "Thời gian xuất kho đơn hàng {subject} tối đa {hours} giờ kể từ khi xác nhận.",
            ], hours=[4, 12, 24, 48]),
            _c("inner_city", [
                "Đơn hàng nội thành được giao trong {days} ngày.",
                "Thời gian giao đơn hàng nội thành là {days} ngày.",
            ], days=[1, 2, 3]),
            _c("free_ship", [
                "Miễn phí giao hàng cho đơn hàng từ {amount} nghìn đồng.",
                "Đơn hàng trị giá từ {amount} nghìn đồng được miễn phí vận chuyển.",
            ], amount=[300, 500, 1000]),
            _c("cod", [
                "Đơn hàng thu tiền khi giao không vượt quá {amount} triệu đồng.",
                "Giá trị tối đa của đơn hàng thanh toán khi nhận hàng là {amount} triệu đồng.",
            ], amount=[5, 10, 20]),
            _c("approver", [
                "Đơn hàng giao gấp phải được {role} phê duyệt.",
                "{role} phê duyệt mọi đơn hàng giao gấp.",
            ], role=["trưởng kho", "quản lý vận hành", "trưởng phòng Kho vận"]),
        ),
    ),
)

POLICY_SUBJECTS = (
    "chi nhánh Hà Nội", "chi nhánh TP. Hồ Chí Minh", "chi nhánh Đà Nẵng", "chi nhánh Cần Thơ",
    "chi nhánh Hải Phòng", "kênh bán hàng trực tuyến", "hệ thống đại lý", "cửa hàng flagship",
    "khối văn phòng", "khối nhà máy", "kênh bán buôn", "khu vực miền Bắc", "khu vực miền Trung",
    "khu vực miền Nam", "chuỗi cửa hàng tiện lợi", "kênh doanh nghiệp", "trung tâm phân phối Bình Dương",
    "trung tâm phân phối Bắc Ninh", "văn phòng đại diện", "chi nhánh Nha Trang", "chi nhánh Huế",
    "chi nhánh Vinh", "chi nhánh Quy Nhơn", "chi nhánh Buôn Ma Thuột", "chi nhánh Biên Hoà",
    "chi nhánh Vũng Tàu", "chi nhánh Thái Nguyên", "chi nhánh Nam Định", "chi nhánh Long Xuyên",
    "kênh bán hàng qua tổng đài", "kênh sàn thương mại điện tử", "khối dịch vụ hậu mãi",
    "trung tâm bảo hành Hà Nội", "trung tâm bảo hành TP. Hồ Chí Minh",
)

POLICY_BOILERPLATE = (
    "Mọi đơn hàng và chứng từ liên quan phải được phê duyệt theo quy trình chung của công ty.",
    "Quy trình này áp dụng cho toàn bộ nhân viên và đơn vị trực thuộc công ty.",
    "Các trường hợp ngoại lệ phải được báo cáo và phê duyệt bằng văn bản theo quy trình.",
)

TECH_BOILERPLATE = (
    "Tài liệu này thuộc bộ quy trình kỹ thuật nội bộ.",
)

TENANTS = {
    "tech": Tenant(
        key="tech",
        name="KB kỹ thuật",
        code_prefix="SOP",
        departments=("Hạ tầng", "Bảo mật", "Phát triển phần mềm", "Vận hành hệ thống", "Dữ liệu", "Hỗ trợ kỹ thuật"),
        subjects=TECH_SUBJECTS,
        topics=TECH_TOPICS,
        boilerplate=TECH_BOILERPLATE,
    ),
    "policy": Tenant(
        key="policy",
        name="KB chính sách",
        code_prefix="POL",
        departments=("Kinh doanh", "Chăm sóc khách hàng", "Nhân sự", "Tài chính", "Mua hàng", "Kho vận"),
        subjects=POLICY_SUBJECTS,
        topics=POLICY_TOPICS,
        boilerplate=POLICY_BOILERPLATE,
    ),
}
