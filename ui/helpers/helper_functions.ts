function formatSessionDate(sessionId: string): string {
    const year = sessionId.substring(0, 4);
    const month = sessionId.substring(4, 6);
    const day = sessionId.substring(6, 8);
    return `${day}/${month}/${year}`;
}

export { formatSessionDate }